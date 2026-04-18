use assert_cmd::Command;
use predicates::prelude::*;

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
