//! End-to-end coverage for private-custody REPLICATE proof provisioning.

use assert_cmd::Command;
use ed25519_dalek::{SigningKey, VerifyingKey};
use tempfile::tempdir;
use triblespace_core::capability::{
    CapabilityClaim, CapabilityMode, CapabilityProofBundle, CapabilityRequest,
};
use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::proof::CapabilityProofStore;
use triblespace_net::replica::{replicate_capability_atom, ReplicaSetId};

fn field(stdout: &[u8], label: &str) -> String {
    std::str::from_utf8(stdout)
        .expect("utf8 output")
        .lines()
        .find_map(|line| line.trim().strip_prefix(label).map(str::trim))
        .unwrap_or_else(|| panic!("missing {label:?} in {}", String::from_utf8_lossy(stdout)))
        .to_owned()
}

fn parse_key(text: &str) -> VerifyingKey {
    let raw: [u8; 32] = hex::decode(text).unwrap().try_into().unwrap();
    VerifyingKey::from_bytes(&raw).unwrap()
}

fn init_network_key(pile: &std::path::Path, key: &std::path::Path) -> String {
    let output = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "signing-key",
            "init",
            pile.to_str().unwrap(),
            "--key",
            key.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    field(&output, "public-key:")
}

#[test]
fn create_issue_and_join_use_one_exact_replica_set() {
    let dir = tempdir().unwrap();
    let first_pile = dir.path().join("first.pile");
    let second_pile = dir.path().join("second.pile");
    std::fs::File::create(&first_pile).unwrap();
    std::fs::File::create(&second_pile).unwrap();
    let first_key = dir.path().join("first.network.key");
    let second_key = dir.path().join("second.network.key");
    let first_subject = init_network_key(&first_pile, &first_key);
    let second_subject = init_network_key(&second_pile, &second_key);
    let replica_root_key = dir.path().join("replica-root.key");
    let first_invite = dir.path().join("first.replica");
    let second_invite = dir.path().join("second.replica");
    let replica_set = "ab".repeat(32);

    let created = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "replica",
            "create",
            "--root-key",
            replica_root_key.to_str().unwrap(),
            "--replica-set",
            &replica_set,
            "--subject",
            &first_subject,
            "--out",
            first_invite.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let replica_root = field(&created, "replica root pubkey:");
    let first_proof = field(&created, "issued proof id:");
    assert_eq!(field(&created, "replica set:"), replica_set);
    assert_eq!(first_proof.len(), 64);

    let first_bundle = CapabilityProofBundle::from_bytes(&std::fs::read(&first_invite).unwrap())
        .expect("portable first invite");
    assert_eq!(first_bundle.proof().step_count(), 1);
    let verified = first_bundle
        .verify(
            parse_key(&replica_root),
            triblespace_core::clock::epoch_now(),
            parse_key(&first_subject),
            CapabilityRequest::new(
                replicate_capability_atom(ReplicaSetId::new([0xab; 32])),
                CapabilityMode::Invoke,
            ),
        )
        .unwrap();
    assert_eq!(verified.effective_mode(), CapabilityMode::Invoke);

    let issued = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "replica",
            "issue",
            "--root-key",
            replica_root_key.to_str().unwrap(),
            "--replica-set",
            &replica_set,
            "--subject",
            &second_subject,
            "--out",
            second_invite.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(field(&issued, "replica root pubkey:"), replica_root);
    assert_eq!(field(&issued, "replica set:"), replica_set);

    let joined = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "replica",
            "join",
            "--pile",
            second_pile.to_str().unwrap(),
            "--network-key",
            second_key.to_str().unwrap(),
            "--replica-root",
            &replica_root,
            "--replica-set",
            &replica_set,
            "--invite",
            second_invite.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let accepted = field(&joined, "accepted proof:");
    assert_eq!(accepted, field(&issued, "issued proof id:"));

    let mut pile = Pile::open(&second_pile).unwrap();
    pile.refresh().unwrap();
    let proof_raw: [u8; 32] = hex::decode(accepted).unwrap().try_into().unwrap();
    assert!(pile
        .proof(triblespace_core::inline::Inline::new(proof_raw))
        .unwrap()
        .is_some());
    pile.close().unwrap();
}

#[test]
fn join_accepts_delegated_ancestry_with_an_invoke_only_leaf() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("node.pile");
    let wrong_pile_path = dir.path().join("wrong.pile");
    std::fs::File::create(&pile_path).unwrap();
    std::fs::File::create(&wrong_pile_path).unwrap();
    let network_key_path = dir.path().join("node.network.key");
    init_network_key(&pile_path, &network_key_path);
    let connect_root_key = dir.path().join("connect-root.key");
    let connect = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "create",
            "--pile",
            pile_path.to_str().unwrap(),
            "--key",
            network_key_path.to_str().unwrap(),
            "--root-key",
            connect_root_key.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let connect_root = field(&connect, "team root pubkey:");
    let connect_proof = field(&connect, "founder proof id:");
    let network_key = triblespace_core::signing_key_file::load_existing(&network_key_path).unwrap();
    let root = SigningKey::from_bytes(&[71; 32]);
    let delegate = SigningKey::from_bytes(&[72; 32]);
    let replica_set = ReplicaSetId::new([73; 32]);
    let atom = replicate_capability_atom(replica_set);
    let parent = CapabilityProofBundle::issue_root(
        &root,
        CapabilityClaim::root(atom, CapabilityMode::InvokeAndDelegate, None),
        delegate.verifying_key(),
    )
    .unwrap();
    let verified_parent = parent
        .verify(
            root.verifying_key(),
            triblespace_core::clock::epoch_now(),
            delegate.verifying_key(),
            CapabilityRequest::new(atom, CapabilityMode::Delegate),
        )
        .unwrap();
    let invite = verified_parent
        .delegate(
            &delegate,
            CapabilityClaim::delegated(
                verified_parent.claim_handle(),
                atom,
                CapabilityMode::Invoke,
                None,
            ),
            network_key.verifying_key(),
        )
        .unwrap();
    let invite_path = dir.path().join("delegated.replica");
    std::fs::write(&invite_path, invite.to_bytes().unwrap()).unwrap();
    let replica_proof = hex::encode(invite.proof().id().raw);
    let root_text = hex::encode(root.verifying_key().to_bytes());
    let set_text = hex::encode(replica_set.into_bytes());

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "replica",
            "join",
            "--pile",
            pile_path.to_str().unwrap(),
            "--network-key",
            network_key_path.to_str().unwrap(),
            "--replica-root",
            &root_text,
            "--replica-set",
            &set_text,
            "--invite",
            invite_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    let receive_temp = dir.path().join("receive-temp");
    std::fs::create_dir(&receive_temp).unwrap();
    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "net",
            "custody",
            "status",
            pile_path.to_str().unwrap(),
            "--network-key",
            network_key_path.to_str().unwrap(),
            "--connect-root",
            &connect_root,
            "--connect-proof",
            &connect_proof,
            "--replica-root",
            &root_text,
            "--replica-set",
            &set_text,
            "--replica-proof",
            &replica_proof,
            "--temp-dir",
            receive_temp.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "authorization:  REPLICATE_STORE accepted",
        ));

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "replica",
            "join",
            "--pile",
            wrong_pile_path.to_str().unwrap(),
            "--network-key",
            network_key_path.to_str().unwrap(),
            "--replica-root",
            &root_text,
            "--replica-set",
            &"ff".repeat(32),
            "--invite",
            invite_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("replica invite rejected"));

    let widened = CapabilityProofBundle::issue_root(
        &root,
        CapabilityClaim::root(atom, CapabilityMode::InvokeAndDelegate, None),
        network_key.verifying_key(),
    )
    .unwrap();
    let widened_path = dir.path().join("widened.replica");
    std::fs::write(&widened_path, widened.to_bytes().unwrap()).unwrap();
    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "replica",
            "join",
            "--pile",
            wrong_pile_path.to_str().unwrap(),
            "--network-key",
            network_key_path.to_str().unwrap(),
            "--replica-root",
            &root_text,
            "--replica-set",
            &set_text,
            "--invite",
            widened_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("invoke-only authority"));
}
