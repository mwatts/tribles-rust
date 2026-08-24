//! End-to-end tests for exact blob-native team credentials.

use assert_cmd::Command;
use tempfile::tempdir;

struct CreatedTeam {
    root: String,
    root_secret: String,
    founder_credential: String,
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
        founder_credential: output_field(&output, "founder credential:"),
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
fn create_invite_join_and_delegate_compose_by_exact_handles() {
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
    assert_eq!(team.founder_credential.len(), 64);

    let founder_show = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "show",
            "--pile",
            founder_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--credential",
            &team.founder_credential,
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let founder_show = String::from_utf8(founder_show).unwrap();
    assert!(founder_show.contains("ancestry:   1 step(s), root to leaf"));
    assert!(founder_show.contains("action:     CONNECT"));
    assert!(founder_show.contains("mode:       invoke+delegate"));
    assert!(founder_show.contains(&format!("resource:   {}", team.root)));

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
            &team.founder_credential,
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
    let invitee_credential = output_field(&invite_output, "issued credential:");
    assert_eq!(invitee_credential.len(), 64);
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
    assert_eq!(
        output_field(&join_output, "accepted credential:"),
        invitee_credential
    );
    assert_eq!(output_field(&join_output, "proof steps:"), "2");

    // Importing the same exact blobs is idempotent; it does not create a
    // second membership or registry occurrence.
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
            &invitee_credential,
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
    let third_credential = output_field(&third_invite, "issued credential:");
    assert_eq!(third_credential.len(), 64);
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
            "--credential",
            &third_credential,
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let show = String::from_utf8(show).unwrap();
    assert!(show.contains("ancestry:   3 step(s), root to leaf"));
    assert!(show.contains("level 0:"));
    assert!(show.contains("level 1:"));
    assert!(show.contains("level 2:"));
    assert!(show.contains("mode:       invoke"));
}

#[test]
fn join_rejects_a_bundle_for_another_key_without_importing_it() {
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

    let invite = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "invite",
            "--pile",
            founder_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--parent",
            &team.founder_credential,
            "--key",
            founder_key.to_str().unwrap(),
            "--invitee",
            &intended,
            "--out",
            bundle.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let credential = output_field(&invite, "issued credential:");

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

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "show",
            "--pile",
            receiver_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--credential",
            &credential,
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("load credential"));
}

#[test]
fn invite_requires_the_exact_resident_delegating_parent_handle() {
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
            &"11".repeat(32),
            "--key",
            founder_key.to_str().unwrap(),
            "--invitee",
            &invitee,
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("requires a missing blob"));
    assert!(!out.exists());
}

#[test]
fn join_rejects_an_expired_child_at_the_explicit_current_time() {
    let dir = tempdir().unwrap();
    let founder_pile = dir.path().join("founder.pile");
    let receiver_pile = dir.path().join("receiver.pile");
    std::fs::File::create(&founder_pile).unwrap();
    std::fs::File::create(&receiver_pile).unwrap();
    let founder_key = dir.path().join("founder.key");
    let receiver_key = dir.path().join("receiver.key");
    let bundle = dir.path().join("expired.team");
    let team = create_team(&founder_pile, &founder_key);
    let receiver = identity(&receiver_key);

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
            &team.founder_credential,
            "--key",
            founder_key.to_str().unwrap(),
            "--invitee",
            &receiver,
            "--valid-from",
            "2000-01-01T00:00:00Z",
            "--valid-until",
            "2001-01-01T00:00:00Z",
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
            receiver_key.to_str().unwrap(),
            "--invite",
            bundle.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("expired"));
}

#[test]
fn invite_rejects_inverted_or_half_specified_validity() {
    let dir = tempdir().unwrap();
    let pile = dir.path().join("team.pile");
    std::fs::File::create(&pile).unwrap();
    let founder_key = dir.path().join("founder.key");
    let invitee_key = dir.path().join("invitee.key");
    let out = dir.path().join("invite.team");
    let team = create_team(&pile, &founder_key);
    let invitee = identity(&invitee_key);
    let base = [
        "team",
        "invite",
        "--pile",
        pile.to_str().unwrap(),
        "--team-root",
        &team.root,
        "--parent",
        &team.founder_credential,
        "--key",
        founder_key.to_str().unwrap(),
        "--invitee",
        &invitee,
    ];

    Command::cargo_bin("trible")
        .unwrap()
        .args(base)
        .args([
            "--valid-from",
            "2030-01-01T00:00:00Z",
            "--valid-until",
            "2029-01-01T00:00:00Z",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("inverted"));
    assert!(!out.exists());

    Command::cargo_bin("trible")
        .unwrap()
        .args(base)
        .args([
            "--valid-from",
            "2030-01-01T00:00:00Z",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--valid-until"));
}

#[test]
fn show_rejects_the_right_credential_under_a_different_team_resource() {
    let dir = tempdir().unwrap();
    let first_pile = dir.path().join("first.pile");
    let second_pile = dir.path().join("second.pile");
    std::fs::File::create(&first_pile).unwrap();
    std::fs::File::create(&second_pile).unwrap();
    let first = create_team(&first_pile, &dir.path().join("first.key"));
    let second = create_team(&second_pile, &dir.path().join("second.key"));

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "show",
            "--pile",
            first_pile.to_str().unwrap(),
            "--team-root",
            &second.root,
            "--credential",
            &first.founder_credential,
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("credential proof rejected"));
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
        vec![0; 32 + triblespace_net::protocol::MAX_CAPABILITY_PROOF_BYTES + 1],
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
fn team_surface_has_no_enumeration_or_retired_workflow() {
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
        "list",
        "list-pending",
        "list-issued",
        "retract",
        "request-join",
        "approve",
    ] {
        assert!(!help.contains(retired), "retired command {retired} remains");
    }
    for current in ["create", "invite", "join", "show"] {
        assert!(
            help.contains(current),
            "current command {current} is missing"
        );
    }
}
