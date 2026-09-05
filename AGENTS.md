# Mandatory TSUBAME policy

Before any TSUBAME SSH, SFTP, SCP, rsync, Ansible, scheduler, or remote-shell
operation, read `REMOTE_AUTOMATION_POLICY.md` completely and enforce it.

The only authorized remote project root is:

`/gs/bs/tgh-26IAW/hongbo/project_4_coauthor`

No agent may weaken or bypass this boundary, follow symlinks outside it, use a
different remote root, expose credentials, run a persistent service, or submit
jobs repeatedly. If an operation cannot be proven compliant before execution,
stop and ask the registered account holder.

