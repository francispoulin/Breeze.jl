"""Publish current commits, then migrate only after benchmark and Nsight completion."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone


CONFIG = json.loads(Path(__file__).with_name("queue.json").read_text())
WORKSPACE = Path(CONFIG["workspace"])
RUNTIME = Path(CONFIG["runtime"])
DRIVER = "benchmarking/distributed/distributed_cbl_benchmark.jl"
ENVIRONMENT = dict(os.environ, GIT_TERMINAL_PROMPT="0")


def save(status, **details):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    value = {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
             "pid": os.getpid(), **details}
    path = RUNTIME / "status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(value), flush=True)


def git(directory, *arguments, data=None, check=True):
    command = ["git", "-c", "credential.helper=", "-c",
               "credential.helper=!" + CONFIG["github_cli"] + " auth git-credential",
               "-C", str(directory), *arguments]
    return subprocess.run(command, input=data, capture_output=True, env=ENVIRONMENT, check=check)


def text(directory, *arguments):
    return git(directory, *arguments).stdout.decode().strip()


def github_ready():
    result = subprocess.run([CONFIG["github_cli"], "auth", "status", "--hostname", "github.com"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ENVIRONMENT)
    return result.returncode == 0


def push_snapshot():
    branches = (CONFIG["source_branch"], CONFIG["profiling_branch"])
    expected = {branch: text(WORKSPACE, "rev-parse", branch) for branch in branches}
    result = git(WORKSPACE, "push", "--atomic", CONFIG["personal_fork"],
                 *(f"{expected[branch]}:refs/heads/{branch}" for branch in branches), check=False)
    if result.returncode:
        save("WAITING_FOR_GITHUB_PUSH", error=result.stderr.decode().strip())
        return None
    remote = dict(line.split()[::-1] for line in text(WORKSPACE, "ls-remote", "--heads", CONFIG["personal_fork"],
                                                    *(f"refs/heads/{branch}" for branch in branches)).splitlines())
    assert all(remote[f"refs/heads/{branch}"] == expected[branch] for branch in branches)
    return expected[CONFIG["source_branch"]]


def process_running(pid):
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode()
    except FileNotFoundError:
        return False
    return any(name in command for name in ("run_series.py", "report_progress.py", "run_profiles.py"))


def workflows_ready(workspace, alive=process_running):
    studies = (("weak_scaling_mpi_nccl_2026-11-09_dt0p005", 14),
               ("nsight_mpi_nccl_2026-11-09_dt0p005", 6))
    for name, expected in studies:
        directory = workspace / "benchmarking/results/rondeau" / name
        try:
            state = json.loads((directory / "study.json").read_text())
            validation = json.loads((directory / "validation/results.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False, name + ": waiting for final validation"
        if state["status"] != "COMPLETE" or not validation.get("finalized_utc"):
            return False, name + ": " + state["status"]
        if len(state["cases"]) != expected or not all(case["state"] == "COMPLETE" and case.get("fields_finite")
                                                     for case in state["cases"]):
            return False, name + ": numerical completion is not verified"
        if expected == 6 and (validation.get("verified_rank_captures") != 18 or
                              not validation.get("all_capture_sizes_hashes_kernel_counts_and_readme_paths_verified")):
            return False, name + ": rank reports are not fully verified"
        for path in (directory / "execution").glob("*process.json"):
            if alive(json.loads(path.read_text())["pid"]):
                return False, name + ": controller/finalizer still running"
    return True, "Both workflows complete and all controllers stopped"


def merged_driver(source_bytes):
    assert hashlib.sha256(source_bytes).hexdigest() == CONFIG["profiling_driver_sha256"]
    source = source_bytes.decode()
    original = 'comm_name     = uppercase(option("comm", "MPI"))'
    assert source.count(original) == 1
    source = source.replace(original, 'comm_name     = uppercase(option("comm", option("communicator", "MPI")))')
    anchor = 'comm_name in ("MPI", "NCCL") || error("--comm must be MPI or NCCL")'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + '\ncomm_name == uppercase(option("communicator", comm_name)) ||\n'
                            '    error("--comm and --communicator must agree when both are supplied")')
    return source.encode()


def migrate():
    assert text(WORKSPACE, "branch", "--show-current") == CONFIG["source_branch"]
    assert not text(WORKSPACE, "status", "--porcelain"), "Uncommitted work remains; preserve and review before migration"
    assert not text(WORKSPACE, "diff", "--cached", "--name-only")
    staging = Path(CONFIG["staging_clone"])
    archive = Path(CONFIG["archived_checkout"])
    assert not staging.exists() and not archive.exists(), "A staging clone or backup already exists; review it before retrying"
    save("COMBINING_PROFILING_CHANGES")
    git(WORKSPACE, "cherry-pick", CONFIG["profiling_commit"])
    assert push_snapshot(), "Final source push must succeed before cloning or moving anything"
    source_head = text(WORKSPACE, "rev-parse", "HEAD")
    git(WORKSPACE, "bundle", "create", str(RUNTIME / "complete-source-backup.bundle"), "--all")
    save("CLONING_TARGET", source_commit=source_head)
    subprocess.run(["git", "clone", "--origin", "origin", CONFIG["target_fork"], str(staging)],
                   capture_output=True, env=ENVIRONMENT, check=True)
    git(staging, "switch", "-c", CONFIG["new_branch"], "origin/" + CONFIG["target_base_branch"])
    base = text(staging, "rev-parse", "HEAD")
    assert hashlib.sha256((staging / DRIVER).read_bytes()).hexdigest() == CONFIG["reviewed_target_driver_sha256"], \
        "The fork driver changed since review; integrate its newer changes before continuing"
    git(staging, "fetch", str(WORKSPACE), "+refs/heads/*:refs/remotes/previous/*")
    git(staging, "remote", "add", "personal", CONFIG["personal_fork"])
    git(staging, "remote", "add", "previous", str(archive))
    patch = git(WORKSPACE, "diff", "--binary", "--full-index", CONFIG["imported_tree"], source_head,
                "--", ".gitignore", "AGENTS.md", "todo.txt", "benchmarking", ":(exclude)" + DRIVER).stdout
    (RUNTIME / "transfer.patch").write_bytes(patch)
    save("APPLYING_BENCHMARK_CHANGES", target_base=base, source_commit=source_head, staging_clone=str(staging))
    result = git(staging, "apply", "--3way", "--index", str(RUNTIME / "transfer.patch"), check=False)
    if result.returncode:
        (RUNTIME / "integration_errors.log").write_bytes(result.stderr)
        raise RuntimeError("Transfer has conflicts in the staging clone; original checkout and target base are preserved")
    (staging / DRIVER).write_bytes(merged_driver(git(WORKSPACE, "show", source_head + ":" + DRIVER).stdout))
    git(staging, "add", "--", DRIVER)
    assert not text(staging, "diff", "--name-only", base, "--", "src", "ext", "Project.toml"), \
        "Migration must preserve the fork's scientific code and root dependencies"
    git(staging, "diff", "--cached", "--check")
    subprocess.run(["/u/ajtse/software/julia-1.11.9/bin/julia", "--startup-file=no", "-e",
                    'Meta.parseall(read(ARGS[1], String)); println("MIGRATED_DRIVER_PARSE_OK")', str(staging / DRIVER)],
                   capture_output=True, check=True)
    receipt = {"source_commit": source_head, "target_base": base, "branch": CONFIG["new_branch"],
               "origin": CONFIG["target_fork"], "personal_backup": CONFIG["personal_fork"],
               "archived_checkout": str(archive), "raw_nsight_profiles_unchanged": True,
               "historical_results": "Measurements describe their recorded pre-migration source/environment, not the new fork version",
               "driver_interface": "Both --comm and --communicator accepted; inconsistent values rejected"}
    (staging / "benchmarking/queued_fork_migration/migration_result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    git(staging, "add", "--", "benchmarking/queued_fork_migration/migration_result.json")
    git(staging, "commit", "-m", "Carry benchmark work and results onto fjp/distributed-benchmark")
    git(staging, "push", "--set-upstream", "personal", "HEAD:refs/heads/" + CONFIG["new_branch"])
    head = text(staging, "rev-parse", "HEAD")
    assert text(staging, "ls-remote", CONFIG["personal_fork"], "refs/heads/" + CONFIG["new_branch"]).split()[0] == head
    assert not text(staging, "status", "--porcelain")
    save("PROMOTING_NEW_CHECKOUT", target_commit=head, backup=str(archive))
    WORKSPACE.rename(archive)
    try:
        git(archive, "worktree", "repair", CONFIG["profiling_worktree"])
        staging.rename(WORKSPACE)
    except Exception:
        if not WORKSPACE.exists():
            archive.rename(WORKSPACE)
            git(WORKSPACE, "worktree", "repair", CONFIG["profiling_worktree"], check=False)
        raise
    assert text(WORKSPACE, "branch", "--show-current") == CONFIG["new_branch"]
    assert text(WORKSPACE, "remote", "get-url", "origin") == CONFIG["target_fork"]
    save("COMPLETE", workspace=str(WORKSPACE), branch=CONFIG["new_branch"], commit=head, archived_checkout=str(archive))


def main():
    last_pushed = None
    last_wait = None
    while True:
        authenticated = github_ready()
        head = text(WORKSPACE, "rev-parse", "HEAD")
        if authenticated and head != last_pushed:
            pushed = push_snapshot()
            if pushed:
                last_pushed = pushed
                save("SOURCE_PUSHED", commit=pushed, personal_fork=CONFIG["personal_fork"])
        ready, reason = workflows_ready(WORKSPACE)
        if ready and last_pushed == head:
            break
        wait = (ready, reason, authenticated)
        if wait != last_wait:
            save("WAITING_FOR_WORKFLOWS" if authenticated else "WAITING_FOR_GITHUB_AUTH", reason=reason,
                 personal_fork=CONFIG["personal_fork"])
            last_wait = wait
        time.sleep(CONFIG["poll_seconds"])
    try:
        migrate()
    except Exception as error:
        if isinstance(error, subprocess.CalledProcessError):
            detail = ((error.stderr or b"") + (error.stdout or b"")).decode(errors="replace")
        else:
            detail = str(error)
        save("NEEDS_REVIEW", error=detail, staging_clone=CONFIG["staging_clone"], archived_checkout=CONFIG["archived_checkout"])
        raise


if __name__ == "__main__":
    main()
