import argparse
import hashlib
import json
import os
import sqlite3
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path


CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB


@dataclass(frozen=True)
class FileInfo:
    relpath: str
    size: int
    mtime_ns: int
    mode: int
    blob_hash: str


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
          k TEXT PRIMARY KEY,
          v TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS blobs (
          hash TEXT PRIMARY KEY,
          size INTEGER NOT NULL,
          chunk_size INTEGER NOT NULL,
          chunks INTEGER NOT NULL,
          created_ts REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS blob_chunks (
          hash TEXT NOT NULL,
          idx INTEGER NOT NULL,
          data BLOB NOT NULL,
          PRIMARY KEY (hash, idx),
          FOREIGN KEY (hash) REFERENCES blobs(hash) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS file_state (
          path TEXT PRIMARY KEY,
          size INTEGER NOT NULL,
          mtime_ns INTEGER NOT NULL,
          mode INTEGER NOT NULL,
          hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS commits (
          commit_id TEXT PRIMARY KEY,
          parent TEXT,
          ts REAL NOT NULL,
          message TEXT NOT NULL,
          tree TEXT NOT NULL,
          root TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tree_entries (
          tree TEXT NOT NULL,
          path TEXT NOT NULL,
          hash TEXT NOT NULL,
          size INTEGER NOT NULL,
          mtime_ns INTEGER NOT NULL,
          mode INTEGER NOT NULL,
          PRIMARY KEY (tree, path)
        );

        CREATE INDEX IF NOT EXISTS idx_tree_entries_tree ON tree_entries(tree);
        """
    )
    conn.execute("INSERT OR IGNORE INTO meta(k, v) VALUES('schema', '1')")
    conn.commit()


def _fmt_bytes(n: int):
    n = int(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1024.0:.1f}{unit}"
        n //= 1024


def _default_excludes(root: Path, db_path: Path):
    db_path = db_path.resolve()
    root = root.resolve()

    excludes = {
        str((root / ".git").resolve()),
        str(db_path),
        str((db_path.parent / (db_path.name + "-wal")).resolve()),
        str((db_path.parent / (db_path.name + "-shm")).resolve()),
    }
    return excludes


def _iter_files(root: Path, excludes_abs: set[str]):
    for p in root.rglob("*"):
        try:
            rp = p.resolve()
        except OSError:
            continue
        rp_str = str(rp)

        if any(rp_str == ex or rp_str.startswith(ex + os.sep) for ex in excludes_abs):
            continue
        if p.is_dir():
            continue
        yield p


def _hash_file_stream(p: Path):
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as f:
        while True:
            b = f.read(CHUNK_SIZE)
            if not b:
                break
            size += len(b)
            h.update(b)
    return h.hexdigest(), size


def _ensure_blob(conn: sqlite3.Connection, blob_hash: str, p: Path, size: int):
    cur = conn.execute("SELECT 1 FROM blobs WHERE hash = ?", (blob_hash,))
    if cur.fetchone():
        return

    created_ts = time.time()
    # Insert the parent row first so blob_chunks' FK is satisfied.
    conn.execute(
        "INSERT INTO blobs(hash, size, chunk_size, chunks, created_ts) VALUES(?, ?, ?, ?, ?)",
        (blob_hash, int(size), int(CHUNK_SIZE), 0, float(created_ts)),
    )

    idx = 0
    with p.open("rb") as f:
        while True:
            b = f.read(CHUNK_SIZE)
            if not b:
                break
            conn.execute(
                "INSERT INTO blob_chunks(hash, idx, data) VALUES(?, ?, ?)",
                (blob_hash, idx, sqlite3.Binary(b)),
            )
            idx += 1
    conn.execute("UPDATE blobs SET chunks = ? WHERE hash = ?", (int(idx), blob_hash))


def _load_file_state(conn: sqlite3.Connection):
    st = {}
    for row in conn.execute("SELECT path, size, mtime_ns, mode, hash FROM file_state"):
        st[row[0]] = (int(row[1]), int(row[2]), int(row[3]), str(row[4]))
    return st


def _save_file_state(conn: sqlite3.Connection, info: FileInfo):
    conn.execute(
        "INSERT INTO file_state(path, size, mtime_ns, mode, hash) VALUES(?, ?, ?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime_ns=excluded.mtime_ns, "
        "mode=excluded.mode, hash=excluded.hash",
        (info.relpath, info.size, info.mtime_ns, info.mode, info.blob_hash),
    )


def _compute_tree_id(files: list[FileInfo]):
    h = hashlib.sha256()
    for fi in sorted(files, key=lambda x: x.relpath):
        h.update(fi.relpath.encode("utf-8"))
        h.update(b"\0")
        h.update(fi.blob_hash.encode("ascii"))
        h.update(b"\0")
        h.update(str(fi.size).encode("ascii"))
        h.update(b"\0")
        h.update(str(fi.mtime_ns).encode("ascii"))
        h.update(b"\0")
        h.update(str(fi.mode).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _compute_commit_id(parent: str | None, tree: str, ts: float, message: str):
    h = hashlib.sha256()
    h.update((parent or "").encode("ascii"))
    h.update(b"\0")
    h.update(tree.encode("ascii"))
    h.update(b"\0")
    h.update(f"{ts:.6f}".encode("ascii"))
    h.update(b"\0")
    h.update(message.encode("utf-8"))
    return h.hexdigest()


def _get_head(conn: sqlite3.Connection):
    cur = conn.execute("SELECT v FROM meta WHERE k = 'HEAD'")
    row = cur.fetchone()
    return row[0] if row else None


def _set_head(conn: sqlite3.Connection, commit_id: str):
    conn.execute(
        "INSERT INTO meta(k, v) VALUES('HEAD', ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (commit_id,),
    )


def snapshot(db_path: Path, message: str, root: Path | None = None, include_dot_git: bool = False):
    root = (root or _repo_root()).resolve()
    db_path = db_path.resolve()

    with _connect(db_path) as conn:
        _init_schema(conn)

        parent = _get_head(conn)
        file_state = _load_file_state(conn)

        excludes_abs = _default_excludes(root, db_path)
        if include_dot_git:
            excludes_abs = {ex for ex in excludes_abs if not ex.endswith(os.sep + ".git") and not ex.endswith("\\.git")}

        files: list[FileInfo] = []
        total_bytes = 0
        scanned = 0
        reused = 0

        t0 = time.time()
        conn.execute("BEGIN")
        try:
            for p in _iter_files(root, excludes_abs):
                rel = str(p.resolve().relative_to(root)).replace("\\", "/")
                st = p.stat()
                mode = stat.S_IMODE(st.st_mode)
                size = int(st.st_size)
                mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
                scanned += 1

                prev = file_state.get(rel)
                if prev and prev[0] == size and prev[1] == mtime_ns and prev[2] == mode:
                    blob_hash = prev[3]
                    reused += 1
                else:
                    blob_hash, _ = _hash_file_stream(p)
                    _ensure_blob(conn, blob_hash, p, size=size)

                fi = FileInfo(relpath=rel, size=size, mtime_ns=mtime_ns, mode=mode, blob_hash=blob_hash)
                files.append(fi)
                total_bytes += size
                _save_file_state(conn, fi)

            tree_id = _compute_tree_id(files)
            for fi in files:
                conn.execute(
                    "INSERT OR IGNORE INTO tree_entries(tree, path, hash, size, mtime_ns, mode) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    (tree_id, fi.relpath, fi.blob_hash, fi.size, fi.mtime_ns, fi.mode),
                )

            ts = time.time()
            commit_id = _compute_commit_id(parent=parent, tree=tree_id, ts=ts, message=message)
            conn.execute(
                "INSERT INTO commits(commit_id, parent, ts, message, tree, root) VALUES(?, ?, ?, ?, ?, ?)",
                (commit_id, parent, ts, message, tree_id, str(root)),
            )
            _set_head(conn, commit_id)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    dt = time.time() - t0
    print(
        json.dumps(
            {
                "ok": True,
                "commit": commit_id,
                "parent": parent,
                "files": len(files),
                "bytes": total_bytes,
                "scanned": scanned,
                "reused_state": reused,
                "secs": round(dt, 2),
                "db": str(db_path),
            }
        )
    )
    return commit_id


def list_commits(db_path: Path, n: int = 20):
    with _connect(db_path.resolve()) as conn:
        _init_schema(conn)
        head = _get_head(conn)
        rows = list(
            conn.execute(
                "SELECT commit_id, parent, ts, message, tree FROM commits ORDER BY ts DESC LIMIT ?",
                (int(n),),
            )
        )
    out = []
    for c, p, ts, msg, tree in rows:
        out.append(
            {
                "commit": c,
                "parent": p,
                "ts": ts,
                "message": msg,
                "tree": tree,
                "head": (c == head),
            }
        )
    print(json.dumps(out, indent=2))


def show_commit(db_path: Path, commit: str | None):
    with _connect(db_path.resolve()) as conn:
        _init_schema(conn)
        if commit is None or commit == "HEAD":
            commit = _get_head(conn)
        if not commit:
            raise SystemExit("No commits in backup.")
        row = conn.execute(
            "SELECT commit_id, parent, ts, message, tree, root FROM commits WHERE commit_id = ?",
            (commit,),
        ).fetchone()
        if not row:
            raise SystemExit(f"Unknown commit: {commit}")
        commit, parent, ts, msg, tree, root = row
        files = conn.execute("SELECT COUNT(1) FROM tree_entries WHERE tree = ?", (tree,)).fetchone()[0]
        bytes_sum = conn.execute("SELECT SUM(size) FROM tree_entries WHERE tree = ?", (tree,)).fetchone()[0] or 0
    print(
        json.dumps(
            {
                "commit": commit,
                "parent": parent,
                "ts": ts,
                "message": msg,
                "tree": tree,
                "root": root,
                "files": int(files),
                "bytes": int(bytes_sum),
            },
            indent=2,
        )
    )


def restore(db_path: Path, dest: Path, commit: str | None):
    db_path = db_path.resolve()
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with _connect(db_path) as conn:
        _init_schema(conn)
        if commit is None or commit == "HEAD":
            commit = _get_head(conn)
        if not commit:
            raise SystemExit("No commits in backup.")
        row = conn.execute("SELECT tree FROM commits WHERE commit_id = ?", (commit,)).fetchone()
        if not row:
            raise SystemExit(f"Unknown commit: {commit}")
        tree = row[0]

        t0 = time.time()
        n_files = 0
        bytes_written = 0

        for path, blob_hash, size, mtime_ns, mode in conn.execute(
            "SELECT path, hash, size, mtime_ns, mode FROM tree_entries WHERE tree = ? ORDER BY path",
            (tree,),
        ):
            out_path = dest / Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                for _, data in conn.execute(
                    "SELECT idx, data FROM blob_chunks WHERE hash = ? ORDER BY idx",
                    (blob_hash,),
                ):
                    f.write(data)
            try:
                os.chmod(out_path, int(mode))
            except OSError:
                pass
            try:
                os.utime(out_path, ns=(int(mtime_ns), int(mtime_ns)))
            except OSError:
                pass
            n_files += 1
            bytes_written += int(size)

    dt = time.time() - t0
    print(
        json.dumps(
            {
                "ok": True,
                "commit": commit,
                "dest": str(dest),
                "files": n_files,
                "bytes": bytes_written,
                "secs": round(dt, 2),
            }
        )
    )


def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="One-file, git-like incremental backup (SQLite).")
    p.add_argument("--db", default=str(_repo_root() / "backup" / "backup.db"))

    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("snapshot", help="Snapshot current working tree into the DB.")
    ps.add_argument("-m", "--message", default="snapshot")
    ps.add_argument("--root", default=None, help="Override repo root to snapshot.")
    ps.add_argument("--include-dot-git", action="store_true", help="Include .git directory in snapshot.")

    pl = sub.add_parser("list", help="List recent snapshots.")
    pl.add_argument("-n", type=int, default=20)

    pc = sub.add_parser("show", help="Show one snapshot.")
    pc.add_argument("commit", nargs="?", default="HEAD")

    pr = sub.add_parser("restore", help="Restore snapshot into a directory.")
    pr.add_argument("--to", required=True)
    pr.add_argument("commit", nargs="?", default="HEAD")

    args = p.parse_args(argv)
    db_path = Path(args.db)

    if args.cmd == "snapshot":
        root = Path(args.root).resolve() if args.root else None
        snapshot(db_path=db_path, message=args.message, root=root, include_dot_git=bool(args.include_dot_git))
    elif args.cmd == "list":
        list_commits(db_path=db_path, n=int(args.n))
    elif args.cmd == "show":
        show_commit(db_path=db_path, commit=args.commit)
    elif args.cmd == "restore":
        restore(db_path=db_path, dest=Path(args.to), commit=args.commit)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
