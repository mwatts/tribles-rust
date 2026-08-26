//! End-to-end coverage for explicit private-custody configuration.

use assert_cmd::Command;
use iroh_base::{EndpointAddr, EndpointId, SecretKey, TransportAddr};
use iroh_tickets::endpoint::EndpointTicket;
use tempfile::tempdir;

fn field(stdout: &[u8], label: &str) -> String {
    std::str::from_utf8(stdout)
        .expect("utf8 output")
        .lines()
        .find_map(|line| line.trim().strip_prefix(label).map(str::trim))
        .unwrap_or_else(|| panic!("missing {label:?} in {}", String::from_utf8_lossy(stdout)))
        .to_owned()
}

#[test]
fn status_validates_both_exact_proofs_and_prints_the_static_ticket() {
    let dir = tempdir().unwrap();
    let pile = dir.path().join("custody.pile");
    let network_key = dir.path().join("network.key");
    let connect_root_key = dir.path().join("connect-root.key");
    let replica_root_key = dir.path().join("replica-root.key");
    let replica_invite = dir.path().join("replica.invite");
    let receive_temp = dir.path().join("receive-temp");
    std::fs::File::create(&pile).unwrap();
    std::fs::create_dir(&receive_temp).unwrap();

    let connect = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "create",
            "--pile",
            pile.to_str().unwrap(),
            "--key",
            network_key.to_str().unwrap(),
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

    let key = triblespace_core::signing_key_file::load_existing(&network_key).unwrap();
    let subject = hex::encode(key.verifying_key().to_bytes());
    let replica_set = "a7".repeat(32);
    let replica = Command::cargo_bin("trible")
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
            &subject,
            "--out",
            replica_invite.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let replica_root = field(&replica, "replica root pubkey:");
    let replica_proof = field(&replica, "issued proof id:");

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "team",
            "replica",
            "join",
            "--pile",
            pile.to_str().unwrap(),
            "--network-key",
            network_key.to_str().unwrap(),
            "--replica-root",
            &replica_root,
            "--replica-set",
            &replica_set,
            "--invite",
            replica_invite.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Deliberately unreachable from the loopback-only runtime below. This
    // covers cancellation while iroh/noq still owns a pre-handshake
    // connection, including its EADDRNOTAVAIL path.
    let remote = EndpointId::from(SecretKey::from_bytes(&[93; 32]).public());
    let remote_addr = EndpointAddr::from_parts(
        remote,
        [TransportAddr::Ip("10.242.0.2:49152".parse().unwrap())],
    );
    let remote_ticket = EndpointTicket::new(remote_addr).to_string();
    let status = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "net",
            "custody",
            "status",
            pile.to_str().unwrap(),
            "--network-key",
            network_key.to_str().unwrap(),
            "--bind",
            "10.242.0.1:49152",
            "--peer",
            &remote_ticket,
            "--connect-root",
            &connect_root,
            "--connect-proof",
            &connect_proof,
            "--replica-root",
            &replica_root,
            "--replica-set",
            &replica_set,
            "--replica-proof",
            &replica_proof,
            "--temp-dir",
            receive_temp.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let status_text = String::from_utf8(status.clone()).unwrap();
    assert_eq!(field(&status, "state:"), "configured");
    assert_eq!(field(&status, "bind:"), "10.242.0.1:49152");
    assert_eq!(field(&status, "connect_root:"), connect_root);
    assert_eq!(field(&status, "connect_proof:"), connect_proof);
    assert_eq!(field(&status, "replica_root:"), replica_root);
    assert_eq!(field(&status, "replica_set:"), replica_set);
    assert_eq!(field(&status, "replica_proof:"), replica_proof);
    assert_eq!(field(&status, "peers:"), "1");
    assert_eq!(
        field(&status, "temp_dir:"),
        receive_temp.display().to_string()
    );
    let local_address: EndpointAddr = field(&status, "ticket:")
        .parse::<EndpointTicket>()
        .unwrap()
        .into();
    assert_eq!(
        local_address.ip_addrs().copied().collect::<Vec<_>>(),
        vec!["10.242.0.1:49152".parse::<std::net::SocketAddr>().unwrap()]
    );
    assert!(field(&status, "inventory_blobs:").parse::<u64>().unwrap() >= 2);
    assert!(
        field(&status, "inventory_blob_bytes:")
            .parse::<u64>()
            .unwrap()
            > 0
    );
    assert_eq!(field(&status, "inventory_records:"), "0");
    assert_eq!(field(&status, "inventory_proofs:"), "2");
    assert!(!field(&status, "inventory_build:").is_empty());
    assert!(status_text.contains("authorization:  CONNECT accepted"));
    assert!(status_text.contains("authorization:  REPLICATE_STORE accepted"));

    #[cfg(unix)]
    {
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let bind = socket.local_addr().unwrap().to_string();
        drop(socket);
        let child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("trible"))
            .args([
                "pile",
                "net",
                "custody",
                "run",
                pile.to_str().unwrap(),
                "--network-key",
                network_key.to_str().unwrap(),
                "--bind",
                &bind,
                "--peer",
                &remote_ticket,
                "--connect-root",
                &connect_root,
                "--connect-proof",
                &connect_proof,
                "--replica-root",
                &replica_root,
                "--replica-set",
                &replica_set,
                "--replica-proof",
                &replica_proof,
                "--temp-dir",
                receive_temp.to_str().unwrap(),
                "--interval",
                "600",
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let signal = std::process::Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status()
            .unwrap();
        assert!(signal.success());
        let shutdown_started = std::time::Instant::now();
        let output = child.wait_with_output().unwrap();
        assert!(
            shutdown_started.elapsed() < std::time::Duration::from_secs(5),
            "custody shutdown waited for an in-flight peer sweep"
        );
        assert!(
            output.status.success(),
            "custody run failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stdout.contains("state:          listening"));
        assert!(stdout.contains(&format!("bind:           {bind}")));
        assert!(stdout.contains("ticket:"));
        assert!(stderr.contains("shutdown requested"));
        assert!(!stderr.contains("Pile dropped without close"));
        assert!(!stderr.contains("Endpoint dropped without calling `Endpoint::close`"));
    }
}

#[test]
fn custody_rejects_discovery_ids_and_non_unicast_binds_before_opening_a_pile() {
    let dir = tempdir().unwrap();
    let key_path = dir.path().join("network.key");
    let receive_temp = dir.path().join("receive-temp");
    std::fs::create_dir(&receive_temp).unwrap();
    triblespace_core::signing_key_file::init(&key_path).unwrap();
    let remote = EndpointId::from(SecretKey::from_bytes(&[94; 32]).public());
    let exact = "00".repeat(32);

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "net",
            "custody",
            "status",
            "unused.pile",
            "--network-key",
            key_path.to_str().unwrap(),
            "--bind",
            "10.242.0.1:49152",
            "--peer",
            &remote.to_string(),
            "--connect-root",
            &exact,
            "--connect-proof",
            &exact,
            "--replica-root",
            &exact,
            "--replica-set",
            &exact,
            "--replica-proof",
            &exact,
            "--temp-dir",
            receive_temp.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("expected an EndpointTicket"));

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "net",
            "custody",
            "status",
            "unused.pile",
            "--network-key",
            key_path.to_str().unwrap(),
            "--bind",
            "0.0.0.0:49152",
            "--connect-root",
            &exact,
            "--connect-proof",
            &exact,
            "--replica-root",
            &exact,
            "--replica-set",
            &exact,
            "--replica-proof",
            &exact,
            "--temp-dir",
            receive_temp.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "not one explicit unicast interface",
        ));
}

#[test]
fn custody_help_exposes_only_explicit_static_configuration() {
    let status = Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "net", "custody", "status", "--help"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let status = String::from_utf8(status).unwrap();
    for flag in [
        "--network-key",
        "--bind",
        "--peer",
        "--connect-root",
        "--connect-proof",
        "--replica-root",
        "--replica-set",
        "--replica-proof",
        "--temp-dir",
    ] {
        assert!(status.contains(flag), "missing {flag} in {status}");
    }
    assert!(status.contains("EndpointTicket"));
    assert!(!status.contains("gossip"));
    assert!(!status.contains("discovery"));

    let run = Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "net", "custody", "run", "--help"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert!(String::from_utf8(run).unwrap().contains("--interval"));
}
