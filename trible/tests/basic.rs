use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn signing_key_init_is_explicit_and_idempotent() {
    let directory = tempfile::tempdir().unwrap();
    let pile = directory.path().join("data.pile");
    let key = directory.path().join("self.key");

    let first = Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "signing-key", "init"])
        .arg(&pile)
        .output()
        .unwrap();
    assert!(first.status.success());
    assert_eq!(std::fs::metadata(&key).unwrap().len(), 64);
    assert!(!pile.exists(), "key provisioning must not create the pile");

    let second = Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "signing-key", "init"])
        .arg(&pile)
        .output()
        .unwrap();
    assert!(second.status.success());
    assert_eq!(second.stdout, first.stdout);
    assert!(String::from_utf8(first.stdout)
        .unwrap()
        .contains("public-key: "));
}

#[test]
fn genid_outputs_id() {
    Command::cargo_bin("trible")
        .unwrap()
        .arg("genid")
        .assert()
        .success()
        .stdout(predicate::str::is_match("^[A-F0-9]{32}\\n$").unwrap());
}

#[test]
fn completion_generates_script() {
    Command::cargo_bin("trible")
        .unwrap()
        .args(["completion", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("_trible()"));
}

#[test]
fn version_flag_prints_crate_version() {
    // Both `--version` and `-V` flags work and print
    // `trible <semver>` (clap's default --version format).
    Command::cargo_bin("trible")
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::is_match("^trible \\d+\\.\\d+\\.\\d+\\n$").unwrap());

    Command::cargo_bin("trible")
        .unwrap()
        .arg("-V")
        .assert()
        .success()
        .stdout(predicate::str::is_match("^trible \\d+\\.\\d+\\.\\d+\\n$").unwrap());
}
