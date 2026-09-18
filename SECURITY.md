# Security policy

## Reporting a vulnerability

Use GitHub's private reporting: **Security → Report a vulnerability** on this
repository. That opens a private advisory only the maintainer can see. Please do not
file security problems as public issues.

Include what you would put in a bug report (see the issue template) plus the impact
you believe it has. You should hear back within a week. Once a fix is available, the
advisory is published with credit to the reporter unless you ask otherwise.

If the problem is in Infinity's engine rather than in this port, it also affects
upstream; the maintainer will coordinate with [infiniflow/infinity](https://github.com/infiniflow/infinity)
and ask you not to disclose until both are fixed.

## What is in scope

- The server binary built or packaged from this repository.
- The scripts under `scripts/apple_silicon/`, `install.sh` and the release workflow,
  which download and execute things.
- The Python SDK as shipped here.

## What you should already know

Infinity has **no authentication, authorization or transport encryption**. Anyone who
can reach ports 23817, 23820 or 5432 can read, write and drop everything. The scripts
in this repository bind `127.0.0.1` for that reason. Exposing the server to a network
without a proxy that enforces access control is a deployment error, not a
vulnerability.

The following are known, public, and tracked in [docs/known-issues.md](docs/known-issues.md);
reports of them are welcome as pull requests rather than advisories:

- Snapshot names are used as file paths without sanitization, so a client with query
  access can write files outside the snapshot directory (blocker 4).
- Several ordinary API calls crash the server process (blockers 1 to 3), which is a
  denial of service from any client.
- Ungrouped aggregates over heavily updated tables can return uninitialized memory
  (blocker 6).

## Supported versions

Only the latest release and `main`. There is no backporting.

## Supply chain

`install.sh` verifies the SHA-256 published beside each release tarball.
`install_test_tools.sh` pins the sqllogictest version and verifies a per-platform
checksum before executing what it downloaded. C++ dependencies are pinned through
the vcpkg baseline in `vcpkg.json`; Python dependencies through `uv.lock`.
