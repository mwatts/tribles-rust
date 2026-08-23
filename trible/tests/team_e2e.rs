//! End-to-end tests for the positive authority team CLI.

use assert_cmd::Command;
use tempfile::tempdir;

struct CreatedTeam {
    root: String,
    root_secret: String,
    founder_grant: String,
}

fn output_field(stdout: &[u8], label: &str) -> String {
    std::str::from_utf8(stdout)
        .expect("utf8 output")
        .lines()
        .find_map(|line| line.trim().strip_prefix(label).map(str::trim))
        .unwrap_or_else(|| panic!("missing {label:?} in {}", String::from_utf8_lossy(stdout)))
        .to_owned()
}

fn create_team(pile: &std::path::Path, key: &std::path::Path) -> CreatedTeam {
    let output = Command::cargo_bin("trible")
        .expect("trible binary")
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
    CreatedTeam {
        root: output_field(&output, "team root pubkey:"),
        root_secret: output_field(&output, "team root SECRET:"),
        founder_grant: output_field(&output, "founder grant:"),
    }
}

fn identity(key: &std::path::Path) -> String {
    let output = Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "net", "identity", "--key", key.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    output_field(&output, "node:")
}

#[test]
fn create_invite_join_and_delegate_compose() {
    let dir = tempdir().unwrap();
    let founder_pile = dir.path().join("founder.pile");
    let invitee_pile = dir.path().join("invitee.pile");
    let third_pile = dir.path().join("third.pile");
    for pile in [&founder_pile, &invitee_pile, &third_pile] {
        std::fs::File::create(pile).unwrap();
    }
    let founder_key = dir.path().join("founder.key");
    let invitee_key = dir.path().join("invitee.key");
    let third_key = dir.path().join("third.key");
    let invitee_bundle = dir.path().join("invitee.team");
    let third_bundle = dir.path().join("third.team");

    let team = create_team(&founder_pile, &founder_key);
    assert_eq!(team.root.len(), 64);
    assert_eq!(team.root_secret.len(), 64);
    assert_eq!(team.founder_grant.len(), 32);

    let founder_list = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "list",
            "--pile",
            founder_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let founder_list = String::from_utf8(founder_list).unwrap();
    assert!(founder_list.contains("accepted grants: 1"));
    assert!(founder_list.contains("action:   CONNECT"));
    assert!(founder_list.contains("delegate: true"));
    assert!(founder_list.contains("diagnostics:     0"));

    let invitee = identity(&invitee_key);
    let invite_output = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "invite",
            "--pile",
            founder_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--parent",
            &team.founder_grant,
            "--key",
            founder_key.to_str().unwrap(),
            "--invitee",
            &invitee,
            "--delegate",
            "--out",
            invitee_bundle.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let invitee_grant = output_field(&invite_output, "issued grant:");
    assert_eq!(invitee_grant.len(), 32);
    assert_eq!(output_field(&invite_output, "proof steps:"), "2");

    let join_output = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            invitee_pile.to_str().unwrap(),
            "--key",
            invitee_key.to_str().unwrap(),
            "--invite",
            invitee_bundle.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(output_field(&join_output, "team root:"), team.root);
    assert_eq!(output_field(&join_output, "accepted grant:"), invitee_grant);
    assert_eq!(output_field(&join_output, "proof steps:"), "2");

    // Import is a set insertion, so replaying the same portable evidence is
    // an idempotent success rather than a second logical membership event.
    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            invitee_pile.to_str().unwrap(),
            "--key",
            invitee_key.to_str().unwrap(),
            "--invite",
            invitee_bundle.to_str().unwrap(),
        ])
        .assert()
        .success();

    let status = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "net",
            "status",
            invitee_pile.to_str().unwrap(),
            "--key",
            invitee_key.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--grant",
            &invitee_grant,
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let status = String::from_utf8(status).unwrap();
    assert!(status.contains("proof_steps: 2"));
    assert!(status.contains("authorization: CONNECT accepted"));

    let third = identity(&third_key);
    let third_invite = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "invite",
            "--pile",
            invitee_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--parent",
            &invitee_grant,
            "--key",
            invitee_key.to_str().unwrap(),
            "--invitee",
            &third,
            "--out",
            third_bundle.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let third_grant = output_field(&third_invite, "issued grant:");
    assert_eq!(output_field(&third_invite, "proof steps:"), "3");

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            third_pile.to_str().unwrap(),
            "--key",
            third_key.to_str().unwrap(),
            "--invite",
            third_bundle.to_str().unwrap(),
        ])
        .assert()
        .success();

    let show = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "show",
            "--pile",
            third_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--grant",
            &third_grant,
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let show = String::from_utf8(show).unwrap();
    assert!(show.contains("ancestry: 3 step(s), root to leaf"));
    assert!(show.contains("level 0:"));
    assert!(show.contains("level 1:"));
    assert!(show.contains("level 2:"));
    assert!(show.contains("delegate: false"));
}

#[test]
fn join_rejects_a_bundle_for_another_key() {
    let dir = tempdir().unwrap();
    let founder_pile = dir.path().join("founder.pile");
    let receiver_pile = dir.path().join("receiver.pile");
    std::fs::File::create(&founder_pile).unwrap();
    std::fs::File::create(&receiver_pile).unwrap();
    let founder_key = dir.path().join("founder.key");
    let intended_key = dir.path().join("intended.key");
    let wrong_key = dir.path().join("wrong.key");
    let bundle = dir.path().join("invite.team");
    let team = create_team(&founder_pile, &founder_key);
    let intended = identity(&intended_key);
    let _ = identity(&wrong_key);

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "invite",
            "--pile",
            founder_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--parent",
            &team.founder_grant,
            "--key",
            founder_key.to_str().unwrap(),
            "--invitee",
            &intended,
            "--out",
            bundle.to_str().unwrap(),
        ])
        .assert()
        .success();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            receiver_pile.to_str().unwrap(),
            "--key",
            wrong_key.to_str().unwrap(),
            "--invite",
            bundle.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("invite proof rejected"));

    let list = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "list",
            "--pile",
            receiver_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert!(String::from_utf8(list)
        .unwrap()
        .contains("accepted grants: 0"));
}

#[test]
fn invite_requires_the_exact_accepted_delegating_parent() {
    let dir = tempdir().unwrap();
    let pile = dir.path().join("team.pile");
    std::fs::File::create(&pile).unwrap();
    let founder_key = dir.path().join("founder.key");
    let invitee_key = dir.path().join("invitee.key");
    let out = dir.path().join("invite.team");
    let team = create_team(&pile, &founder_key);
    let invitee = identity(&invitee_key);

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "invite",
            "--pile",
            pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--parent",
            "11111111111111111111111111111111",
            "--key",
            founder_key.to_str().unwrap(),
            "--invitee",
            &invitee,
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("is not accepted"));
    assert!(!out.exists());
}

#[test]
fn join_rejects_an_oversized_bundle_before_decoding() {
    let dir = tempdir().unwrap();
    let pile = dir.path().join("receiver.pile");
    let key = dir.path().join("receiver.key");
    let bundle = dir.path().join("oversized.team");
    std::fs::File::create(&pile).unwrap();
    let _ = identity(&key);
    std::fs::write(
        &bundle,
        vec![0; 32 + triblespace_net::protocol::MAX_AUTHORITY_PROOF_BYTES + 1],
    )
    .unwrap();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            pile.to_str().unwrap(),
            "--key",
            key.to_str().unwrap(),
            "--invite",
            bundle.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("invite bundle exceeds"));
}

#[test]
fn retired_capability_commands_are_absent() {
    let help = Command::cargo_bin("trible")
        .unwrap()
        .args(["team", "--help"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let help = String::from_utf8(help).unwrap();
    for retired in [
        "list-pending",
        "list-issued",
        "retract",
        "request-join",
        "approve",
    ] {
        assert!(!help.contains(retired), "retired command {retired} remains");
    }
    for current in ["create", "invite", "join", "list", "show"] {
        assert!(
            help.contains(current),
            "current command {current} is missing"
        );
    }
}
