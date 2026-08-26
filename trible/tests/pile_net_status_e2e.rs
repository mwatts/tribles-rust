//! End-to-end tests for explicit `pile net` capability configuration.

use assert_cmd::Command;
use tempfile::tempdir;

fn field(stdout: &[u8], label: &str) -> String {
    std::str::from_utf8(stdout)
        .unwrap()
        .lines()
        .find_map(|line| line.trim().strip_prefix(label).map(str::trim))
        .unwrap()
        .to_owned()
}

#[test]
fn status_loads_both_exact_local_team_proofs() {
    let dir = tempdir().unwrap();
    let pile = dir.path().join("team.pile");
    let key = dir.path().join("node.key");
    std::fs::File::create(&pile).unwrap();

    let create = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "create",
            "--pile",
            pile.to_str().unwrap(),
            "--key",
            key.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let root = field(&create, "team root pubkey:");
    let connect_proof = field(&create, "founder connect proof:");
    let sync_proof = field(&create, "founder sync proof:");

    let status = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "net",
            "status",
            pile.to_str().unwrap(),
            "--key",
            key.to_str().unwrap(),
            "--team-root",
            &root,
            "--connect-proof",
            &connect_proof,
            "--sync-proof",
            &sync_proof,
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let status = String::from_utf8(status).unwrap();
    assert!(status.contains(&format!("team_root:           {root}")));
    assert!(status.contains(&format!("connect_proof_id:     {connect_proof}")));
    assert!(status.contains(&format!("sync_proof_id:        {sync_proof}")));
    assert!(status.contains("connect_proof_steps:  1"));
    assert!(status.contains("sync_proof_steps:     1"));
    assert!(status.contains("authorization:       CONNECT + SYNC_TEAM accepted"));
}

#[test]
fn status_has_no_ambient_or_sentinel_configuration() {
    let help = Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "net", "status", "--help"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let help = String::from_utf8(help).unwrap();
    assert!(help.contains("--team-root"));
    assert!(help.contains("--connect-proof"));
    assert!(help.contains("--sync-proof"));
    assert!(!help.contains("--grant"));
    assert!(!help.contains("TRIBLE_TEAM_ROOT"));
    assert!(!help.contains("TRIBLE_TEAM_CAP"));
    assert!(!help.contains("self_cap"));
}

#[test]
fn sync_derives_team_gossip_and_exposes_only_semantic_qos() {
    let help = Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "net", "sync", "--help"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let help = String::from_utf8(help).unwrap();
    assert!(help.contains("--team-root"));
    assert!(help.contains("--connect-proof"));
    assert!(help.contains("--sync-proof"));
    assert!(help.contains("--direction"));
    assert!(help.contains("--blobs"));
    assert!(!help.contains("--gossip-topic"));
    assert!(!help.contains("--no-lazy"));
    assert!(!help.contains("--reconcile-interval"));
    assert!(!help.contains("--grant"));

    let handle = "00".repeat(32);
    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "net",
            "sync",
            "unused.pile",
            "--team-root",
            &handle,
            "--connect-proof",
            &handle,
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--sync-proof"));
}
