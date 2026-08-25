//! End-to-end tests for native direct team capability proofs.

use assert_cmd::Command;
use ed25519_dalek::VerifyingKey;
use hifitime::Epoch;
use tempfile::tempdir;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::Blob;
use triblespace_core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
    CapabilityProofId, CapabilityRequest, CapabilityResource, CapabilityValidity,
    MAX_CAPABILITY_PROOF_BUNDLE_BYTES,
};
use triblespace_core::id::Id;
use triblespace_core::inline::Inline;
use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::proof::CapabilityProofStore;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut};

struct CreatedTeam {
    root: String,
    root_key: String,
    founder_proof: String,
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
        root_key: output_field(&output, "team root key:"),
        founder_proof: output_field(&output, "founder proof id:"),
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

fn parse_proof_id(text: &str) -> CapabilityProofId {
    let bytes = hex::decode(text).expect("proof id hex");
    let raw: [u8; 32] = bytes.try_into().expect("32-byte proof id");
    Inline::new(raw)
}

fn parse_public_key(text: &str) -> VerifyingKey {
    let bytes = hex::decode(text).expect("public key hex");
    let raw: [u8; 32] = bytes.try_into().expect("32-byte public key");
    VerifyingKey::from_bytes(&raw).expect("valid public key")
}

fn read_invite(path: &std::path::Path) -> CapabilityProofBundle {
    let bytes = std::fs::read(path).expect("read invite");
    CapabilityProofBundle::from_bytes(&bytes).expect("canonical capability proof bundle")
}

fn stored_bundle(
    path: &std::path::Path,
    id: CapabilityProofId,
) -> (usize, Option<CapabilityProofBundle>) {
    let mut pile = Pile::open(path).expect("open pile");
    let proof_count = pile
        .proofs()
        .expect("enumerate native proofs")
        .collect::<Result<Vec<_>, _>>()
        .expect("read native proofs")
        .len();
    let bundle = pile.proof(id).expect("look up native proof").map(|proof| {
        let reader = pile.reader().expect("open claim snapshot");
        let claims = proof
            .claim_handles()
            .map(|handle| {
                let claim: Blob<SimpleArchive> =
                    reader.get(handle).expect("signed claim is resident");
                claim
            })
            .collect();
        CapabilityProofBundle::new(proof, claims)
    });
    pile.close().expect("close pile");
    (proof_count, bundle)
}

fn store_bundle(path: &std::path::Path, bundle: &CapabilityProofBundle) {
    let mut pile = Pile::open(path).expect("open pile");
    for claim in bundle.claims() {
        pile.put::<SimpleArchive, _>(Blob::<SimpleArchive>::new(claim.bytes.clone()))
            .expect("store claim");
    }
    pile.insert_proof(bundle.proof().clone())
        .expect("store native proof");
    pile.close().expect("close pile");
}

fn assert_exact_claim_closure(bundle: &CapabilityProofBundle) {
    assert_eq!(bundle.claims().len(), bundle.proof().step_count());
    assert_eq!(
        bundle.proof().claim_handles().collect::<Vec<_>>(),
        bundle
            .claims()
            .iter()
            .map(Blob::get_handle)
            .collect::<Vec<_>>()
    );
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
    assert_eq!(
        team.root_key,
        founder_pile
            .with_extension("team-root.key")
            .display()
            .to_string()
    );
    assert!(std::path::Path::new(&team.root_key).is_file());
    assert_eq!(team.founder_proof.len(), 64);

    let founder_proof_id = parse_proof_id(&team.founder_proof);
    let (founder_proof_count, founder_bundle) = stored_bundle(&founder_pile, founder_proof_id);
    assert_eq!(founder_proof_count, 1);
    let founder_bundle = founder_bundle.expect("founder proof is stored natively");
    assert_eq!(founder_bundle.proof().id(), founder_proof_id);
    assert_exact_claim_closure(&founder_bundle);

    let founder_show = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "show",
            "--pile",
            founder_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--proof",
            &team.founder_proof,
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let founder_show = String::from_utf8(founder_show).unwrap();
    assert!(founder_show.contains("ancestry:       1 step(s), root to leaf"));
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
            "--parent-proof",
            &team.founder_proof,
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
    let invitee_proof = output_field(&invite_output, "issued proof id:");
    assert_eq!(invitee_proof.len(), 64);
    assert_eq!(output_field(&invite_output, "proof steps:"), "2");

    let invitee_proof_id = parse_proof_id(&invitee_proof);
    let invitee_portable_bundle = read_invite(&invitee_bundle);
    assert_eq!(
        hex::encode(invitee_portable_bundle.proof().root_key().to_bytes()),
        team.root
    );
    assert_eq!(invitee_portable_bundle.proof().id(), invitee_proof_id);
    assert_exact_claim_closure(&invitee_portable_bundle);

    let join_output = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            invitee_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
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
    assert_eq!(output_field(&join_output, "accepted proof:"), invitee_proof);
    assert_eq!(output_field(&join_output, "proof steps:"), "2");

    let (invitee_proof_count, invitee_stored_bundle) =
        stored_bundle(&invitee_pile, invitee_proof_id);
    assert_eq!(invitee_proof_count, 1);
    assert_eq!(
        invitee_stored_bundle.expect("joined proof is stored natively"),
        invitee_portable_bundle
    );

    // Importing the same exact claim closure and proof is idempotent.
    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            invitee_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--key",
            invitee_key.to_str().unwrap(),
            "--invite",
            invitee_bundle.to_str().unwrap(),
        ])
        .assert()
        .success();
    let (invitee_proof_count, invitee_stored_bundle) =
        stored_bundle(&invitee_pile, invitee_proof_id);
    assert_eq!(invitee_proof_count, 1);
    assert_eq!(
        invitee_stored_bundle.expect("idempotent join retains the proof"),
        invitee_portable_bundle
    );

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
            "--parent-proof",
            &invitee_proof,
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
    let third_proof = output_field(&third_invite, "issued proof id:");
    assert_eq!(third_proof.len(), 64);
    assert_eq!(output_field(&third_invite, "proof steps:"), "3");

    let third_proof_id = parse_proof_id(&third_proof);
    let third_portable_bundle = read_invite(&third_bundle);
    assert_eq!(
        third_portable_bundle.proof().root_key(),
        invitee_portable_bundle.proof().root_key()
    );
    assert_eq!(third_portable_bundle.proof().id(), third_proof_id);
    assert_exact_claim_closure(&third_portable_bundle);

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            third_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--key",
            third_key.to_str().unwrap(),
            "--invite",
            third_bundle.to_str().unwrap(),
        ])
        .assert()
        .success();

    let (third_proof_count, third_stored_bundle) = stored_bundle(&third_pile, third_proof_id);
    assert_eq!(third_proof_count, 1);
    assert_eq!(
        third_stored_bundle.expect("delegated proof is stored natively"),
        third_portable_bundle
    );

    let show = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "show",
            "--pile",
            third_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--proof",
            &third_proof,
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let show = String::from_utf8(show).unwrap();
    assert!(show.contains("ancestry:       3 step(s), root to leaf"));
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
            "--parent-proof",
            &team.founder_proof,
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
    let proof = output_field(&invite, "issued proof id:");
    let proof_id = parse_proof_id(&proof);

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            receiver_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--key",
            wrong_key.to_str().unwrap(),
            "--invite",
            bundle.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("invite proof rejected"));

    let (proof_count, stored) = stored_bundle(&receiver_pile, proof_id);
    assert_eq!(proof_count, 0);
    assert!(stored.is_none());

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "show",
            "--pile",
            receiver_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--proof",
            &proof,
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("is not present"));
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
            "--parent-proof",
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
        .stderr(predicates::str::contains("is not present"));
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
    let team_root = parse_public_key(&team.root);
    let receiver = parse_public_key(&receiver);
    let founder = triblespace_core::signing_key_file::load_existing(&founder_key).unwrap();
    let (_, founder_bundle) = stored_bundle(&founder_pile, parse_proof_id(&team.founder_proof));
    let founder_bundle = founder_bundle.expect("resident founder proof");
    let root_claim = CapabilityClaim::from_blob(founder_bundle.claims()[0].clone()).unwrap();
    let valid_from = Epoch::maybe_from_gregorian_utc(2000, 1, 1, 0, 0, 0, 0).unwrap();
    let valid_until = Epoch::maybe_from_gregorian_utc(2001, 1, 1, 0, 0, 0, 0).unwrap();
    let issued_then = founder_bundle
        .verify(
            team_root,
            valid_from,
            founder.verifying_key(),
            CapabilityRequest::new(root_claim.atom(), CapabilityMode::Delegate),
        )
        .unwrap();
    let expired_claim = CapabilityClaim::delegated(
        issued_then.claim_handle(),
        root_claim.atom(),
        CapabilityMode::Invoke,
        Some(CapabilityValidity::new(valid_from, valid_until).unwrap()),
    );
    let expired_bundle = issued_then
        .delegate(&founder, expired_claim, receiver)
        .unwrap();
    let expired_id = expired_bundle.proof().id();
    std::fs::write(&bundle, expired_bundle.to_bytes().unwrap()).unwrap();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            receiver_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--key",
            receiver_key.to_str().unwrap(),
            "--invite",
            bundle.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("expired"));

    let (proof_count, stored) = stored_bundle(&receiver_pile, expired_id);
    assert_eq!(proof_count, 0);
    assert!(stored.is_none());
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
        "--parent-proof",
        &team.founder_proof,
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
fn show_rejects_the_right_proof_under_a_different_team_root() {
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
            "--proof",
            &first.founder_proof,
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("capability proof rejected"));
}

#[test]
fn show_requires_the_exact_connect_atom_even_under_the_right_root() {
    let dir = tempdir().unwrap();
    let pile = dir.path().join("team.pile");
    let founder_key_path = dir.path().join("founder.key");
    std::fs::File::create(&pile).unwrap();
    let team = create_team(&pile, &founder_key_path);
    let root =
        triblespace_core::signing_key_file::load_existing(std::path::Path::new(&team.root_key))
            .unwrap();
    let founder = triblespace_core::signing_key_file::load_existing(&founder_key_path).unwrap();
    let unrelated = CapabilityProofBundle::issue_root(
        &root,
        CapabilityClaim::root(
            CapabilityAtom::new(
                CapabilityAction::new(Id::new([0xA5; 16]).unwrap()),
                CapabilityResource::new([0x5A; 32]),
            ),
            CapabilityMode::Invoke,
            None,
        ),
        founder.verifying_key(),
    )
    .unwrap();
    store_bundle(&pile, &unrelated);

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "show",
            "--pile",
            pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--proof",
            &hex::encode(unrelated.proof().id().raw),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("exact request"));
}

#[test]
fn join_rejects_a_tampered_direct_signature_without_importing_it() {
    let dir = tempdir().unwrap();
    let founder_pile = dir.path().join("founder.pile");
    let receiver_pile = dir.path().join("receiver.pile");
    std::fs::File::create(&founder_pile).unwrap();
    std::fs::File::create(&receiver_pile).unwrap();
    let founder_key = dir.path().join("founder.key");
    let receiver_key = dir.path().join("receiver.key");
    let invite = dir.path().join("tampered.team");
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
            "--parent-proof",
            &team.founder_proof,
            "--key",
            founder_key.to_str().unwrap(),
            "--invitee",
            &receiver,
            "--out",
            invite.to_str().unwrap(),
        ])
        .assert()
        .success();

    let original = read_invite(&invite);
    let mut bytes = std::fs::read(&invite).unwrap();
    // bundle(version, count) || proof(K0, S0, C0, K1, ...)
    let first_signature = 2 + 32;
    bytes[first_signature] ^= 1;
    std::fs::write(&invite, bytes).unwrap();

    let tampered = read_invite(&invite);
    assert_ne!(tampered.proof().id(), original.proof().id());
    assert_exact_claim_closure(&tampered);
    let tampered_id = tampered.proof().id();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            receiver_pile.to_str().unwrap(),
            "--team-root",
            &team.root,
            "--key",
            receiver_key.to_str().unwrap(),
            "--invite",
            invite.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("invalid signature"));

    let (proof_count, stored) = stored_bundle(&receiver_pile, tampered_id);
    assert_eq!(proof_count, 0);
    assert!(stored.is_none());
}

#[test]
fn join_rejects_an_oversized_bundle_before_decoding() {
    let dir = tempdir().unwrap();
    let pile = dir.path().join("receiver.pile");
    let key = dir.path().join("receiver.key");
    let bundle = dir.path().join("oversized.team");
    std::fs::File::create(&pile).unwrap();
    let _ = identity(&key);
    let root = hex::encode(
        ed25519_dalek::SigningKey::from_bytes(&[7; 32])
            .verifying_key()
            .to_bytes(),
    );
    std::fs::write(&bundle, vec![0; MAX_CAPABILITY_PROOF_BUNDLE_BYTES + 1]).unwrap();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "join",
            "--pile",
            pile.to_str().unwrap(),
            "--team-root",
            &root,
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
fn issuance_can_describe_future_authority_without_observing_the_clock() {
    let dir = tempdir().unwrap();
    let pile = dir.path().join("future.pile");
    let key = dir.path().join("future.key");
    std::fs::File::create(&pile).unwrap();

    let output = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "create",
            "--pile",
            pile.to_str().unwrap(),
            "--key",
            key.to_str().unwrap(),
            "--valid-from",
            "2099-01-01T00:00:00Z",
            "--valid-until",
            "2099-12-31T23:59:59Z",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let proof_id = parse_proof_id(&output_field(&output, "founder proof id:"));
    let (proof_count, stored) = stored_bundle(&pile, proof_id);
    assert_eq!(proof_count, 1);
    assert!(stored.is_some());
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
