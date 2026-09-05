# TSUBAME Remote Automation Policy

Policy version: 1

## Authority and identity

- Only the registered TSUBAME account holder may authorize a connection or job.
- Automation may use only that user's already-registered key through a local
  SSH agent or an explicit key path outside the workspace.
- Never request, print, transmit, copy, edit, or store a private key or its
  passphrase. Never create or register another key without explicit approval.
- Never use account sharing, sudo, privilege escalation, another user's files,
  VS Code Remote-SSH, Jupyter Server, cron, a daemon, or a reverse tunnel.
- Institutional approval for an AI-controlled SSH client remains a prerequisite
  for unattended execution. This file does not override TSUBAME, NII, laboratory,
  group-owner, or project rules; the strictest applicable rule wins.

## Immutable remote boundary

The only authorized remote project root is exactly:

`/gs/bs/tgh-26IAW/hongbo/project_4_coauthor`

- All project file creation, modification, deletion, upload, download source,
  extraction, caches, environments, logs, checkpoints, results, temporary
  files, and baseline clones must resolve inside this root.
- Before every mutating operation, resolve the existing nearest parent with
  `realpath -e`, construct the destination beneath it, and reject `..`, an empty
  path, a relative path, shell expansion, wildcard mutation, or a symlink that
  resolves outside the authorized root.
- Never mutate `/home`, `/work`, `/gs/fs`, `/tmp`, another `/gs/bs` location,
  another user's directory, or the authorized root itself.
- Reading system executables/libraries and SSH authentication files is
  unavoidable and permitted; modifying them is not.
- Scheduler state changes are permitted only for the single explicitly
  authorized job. They are not filesystem authorization.

## Login-node limits

Permitted on a login node:

- authenticated file transfer with a conservative bandwidth limit;
- `cd`, `pwd`, `realpath`, `test`, `stat`, `sha256sum`, bounded `find`, bounded
  `du`, `mkdir` below the authorized root, atomic `mv` below the root, and one
  explicit `qsub` submission;
- infrequent status checks and retrieval of small finalized logs/status files.

Forbidden on a login node:

- Python data processing, dataset Map, model loading, training, inference,
  parallel compilation, archive compression/decompression of large payloads,
  or other CPU/memory-heavy work;
- cron, auto-restart loops, persistent servers, high-frequency `qstat`, or
  automatic repeated job submission.

## Deployment and execution protocol

1. Build one local release archive and SHA-256 manifest.
2. Upload to a unique `.uploading` directory below
   `project_4_coauthor/releases`; never overwrite the active release.
3. Verify the archive hash remotely with a lightweight command.
4. Atomically rename the verified release below the same parent directory.
5. Submit exactly one scheduler job. The job must have explicit group, resource,
   GPU type/count, wall time, and non-premium priority unless separately approved.
6. Run all heavy work on allocated compute nodes. Set `TMPDIR`, `HF_HOME`,
   `HF_HUB_CACHE`, `PIP_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, run directories, and
   every other controllable writable cache beneath the authorized root.
7. On failure, stop once, preserve checkpoints, write `FAILURE.json`, and do not
   resubmit automatically.
8. On success, create the result archive outside its source directory, write its
   SHA-256, then create a final completion marker atomically.
9. Download only finalized status/diagnostic/result artifacts. Verify SHA-256
   locally. Never read or copy a checkpoint while it is being written.

## Non-interference requirements

- Never deploy into or modify a directory used by a running job.
- Never signal, inspect private process state of, or alter another job.
- Never kill by broad name, user, shell, or wildcard. A cancellation must use
  the exact scheduler job ID authorized by the user.
- Never use recursive deletion remotely. Quarantine an exact validated path by
  atomic rename within the authorized root when recovery is required.
- Transfer a bounded archive rather than many small files; default bandwidth
  must not exceed the laboratory guide's 50 MB/s example unless approved.
- Do not claim zero impact: network and group-disk I/O exist. Minimize them and
  never transfer large artifacts during a latency-sensitive active run unless
  explicitly approved.

## Required preflight before any connection

Automation must fail closed unless all are true:

- target host is exactly `login.t4.gsic.titech.ac.jp` or an explicitly approved
  official TSUBAME login alias;
- username is supplied by the registered user;
- host-key verification is strict and matches the official/pinned host key;
- the key is outside the workspace and is not group/world readable;
- the remote root string exactly matches the authorized root;
- a read-only remote check proves the root exists, is owned/usable by the
  account, and `realpath -e` equals the authorized root;
- the requested operation has a dry-run plan listing every remote path and any
  scheduler mutation;
- no active release directory will be overwritten;
- the registered user explicitly approves the first real deployment/submission.

## Persistence and change control

- `AGENTS.md` makes this policy mandatory for Codex sessions opened in this
  repository or its parent workspace.
- Keep this policy tracked with the repository and include its SHA-256 in every
  automation run manifest.
- A policy change may tighten restrictions without extra authority. Any change
  that broadens hosts, paths, credentials, deletion, scheduler behavior, or
  unattended control requires explicit approval from the registered user and,
  where applicable, the laboratory/server manager.
