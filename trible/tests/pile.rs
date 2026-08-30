use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;
use triblespace::prelude::BlobStoreList;
use triblespace::prelude::SnapshotSource;
use triblespace_core::repo::pile::Pile;

fn opaque_envelope(needle: Option<[u8; 32]>) -> Vec<u8> {
    let mut record = vec![0u8; 256];
    record[..16].copy_from_slice(
        &hex::decode("E5A95E5D8A0BBA8782E46B9C9E73B313").expect("envelope marker"),
    );
    record[16..32].fill(0xA5);
    record[32..36].copy_from_slice(&1u32.to_le_bytes());
    if let Some(needle) = needle {
        record[80..112].copy_from_slice(&needle);
    }
    record
}

fn legacy_v3_definition_followed_by_blob() -> Vec<u8> {
    const HEADER_LEN: usize = 256;
    let payload = b"prioritize";
    let mut bytes = vec![0u8; 3 * HEADER_LEN];

    bytes[..16].copy_from_slice(
        &hex::decode("3BE108504E4F5242FB24AA72D6D94CE1").expect("definition marker"),
    );
    bytes[16..32].copy_from_slice(&hex::decode("B9566CF892C55CCB0E58411E1B18CD7F").expect("scope"));
    bytes[32..48]
        .copy_from_slice(&hex::decode("8F4A27C8581DADCBA1ADA8BA228069B6").expect("representation"));
    bytes[48..64]
        .copy_from_slice(&hex::decode("6D64C5F4B9E9B73F57C5F8702AB7FE45").expect("recipe"));

    let blob = &mut bytes[HEADER_LEN..];
    blob[..16]
        .copy_from_slice(&hex::decode("9C33EEB525065A62EAEC4BE43DCC355A").expect("V3 blob marker"));
    blob[16..24].copy_from_slice(&1_786_400_694_176u64.to_ne_bytes());
    blob[24..32].copy_from_slice(&(payload.len() as u64).to_ne_bytes());
    blob[32..64].copy_from_slice(blake3::hash(payload).as_bytes());
    blob[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(payload);

    bytes
}

#[test]
fn create_initializes_empty_pile() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("create_test.pile");
    std::fs::File::create(&path).unwrap();

    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "create", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());

    let pile: Pile = Pile::open(&path).unwrap();
    pile.close().unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
}

#[test]
fn create_creates_parent_directories() {
    let dir = tempdir().unwrap();
    let path = dir
        .path()
        .join("nested")
        .join("dirs")
        .join("create_test.pile");

    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "create", path.to_str().unwrap()])
        .assert()
        .success();

    assert!(path.exists());
    assert!(path.parent().unwrap().exists());
}

#[test]
fn put_ingests_file() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("put_test.pile");
    std::fs::File::create(&pile_path).unwrap();
    let input_path = dir.path().join("input.bin");
    std::fs::write(&input_path, b"hello world").unwrap();

    let digest = blake3::hash(b"hello world").to_hex().to_string();
    let handle = format!("blake3:{digest}");
    let pattern = format!("^{handle}\\n$");

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "put",
            pile_path.to_str().unwrap(),
            input_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::is_match(pattern).unwrap());

    let mut pile: Pile = Pile::open(&pile_path).unwrap();
    let snapshot = pile.snapshot().unwrap();
    assert!(snapshot.blobs().next().is_some());
    drop(snapshot);
    pile.close().unwrap();
}

#[test]
fn get_restores_blob() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("get_test.pile");
    std::fs::File::create(&pile_path).unwrap();
    let input_path = dir.path().join("input.bin");
    let output_path = dir.path().join("output.bin");
    let contents = b"fetch me";
    std::fs::write(&input_path, contents).unwrap();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "put",
            pile_path.to_str().unwrap(),
            input_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    let digest = blake3::hash(contents).to_hex().to_string();
    let handle = format!("blake3:{digest}");

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "get",
            pile_path.to_str().unwrap(),
            &handle,
            output_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());

    let out = std::fs::read(&output_path).unwrap();
    assert_eq!(contents, &out[..]);
}

#[test]
fn list_blobs_outputs_expected_handle() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("list_blobs.pile");
    std::fs::File::create(&pile_path).unwrap();
    let input_path = dir.path().join("input.bin");
    let contents = b"hello";
    std::fs::write(&input_path, contents).unwrap();

    let digest = blake3::hash(contents).to_hex().to_string();
    let handle = format!("blake3:{digest}");
    let pattern = format!("^{handle}\\n$");

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "put",
            pile_path.to_str().unwrap(),
            input_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "blob", "list", pile_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::is_match(&pattern).unwrap());
}

#[test]
fn list_blobs_with_metadata_outputs_details() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("list_blobs_meta.pile");
    std::fs::File::create(&pile_path).unwrap();
    let input_path = dir.path().join("input.bin");
    let contents = b"hello";
    std::fs::write(&input_path, contents).unwrap();

    let digest = blake3::hash(contents).to_hex().to_string();
    let handle = format!("blake3:{digest}");
    let pattern = format!(r"^{}\t\S+\t{}\n$", handle, contents.len());

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "put",
            pile_path.to_str().unwrap(),
            input_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "list",
            "--metadata",
            pile_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::is_match(&pattern).unwrap());
}

#[test]
fn diagnose_reports_healthy() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("diag.pile");
    std::fs::File::create(&pile_path).unwrap();

    // create an empty pile file
    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "create", pile_path.to_str().unwrap()])
        .assert()
        .success();

    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "diagnose", "check", pile_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("healthy"));
}

#[test]
fn diagnose_decodes_legacy_v3_definition_and_continues_into_blob() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("legacy-definition-then-blob.pile");
    std::fs::write(&pile_path, legacy_v3_definition_followed_by_blob()).unwrap();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "diagnose",
            "record-at",
            pile_path.to_str().unwrap(),
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "marker: 3BE108504E4F5242FB24AA72D6D94CE1",
        ))
        .stdout(predicate::str::contains(
            "classification: legacy-v3-collection-definition (inert)",
        ))
        .stdout(predicate::str::contains(
            "scope: B9566CF892C55CCB0E58411E1B18CD7F",
        ))
        .stdout(predicate::str::contains(
            "representation: 8F4A27C8581DADCBA1ADA8BA228069B6",
        ))
        .stdout(predicate::str::contains(
            "recipe: 6D64C5F4B9E9B73F57C5F8702AB7FE45",
        ))
        .stdout(predicate::str::contains("known_span_bytes: 256"))
        .stdout(predicate::str::contains("next_offset: 256"));

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "diagnose",
            "record-at",
            pile_path.to_str().unwrap(),
            "256",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("classification: blob"))
        .stdout(predicate::str::contains("known_span_bytes: 512"))
        .stdout(predicate::str::contains("next_offset: 768"))
        .stdout(predicate::str::contains("payload_offset: 512"))
        .stdout(predicate::str::contains("payload_length: 10"))
        .stdout(predicate::str::contains(
            "payload_hash: 15FC745FC8162C584C12017295E065808B04FA51D72EAE20283A2415A4D5B1B0",
        ));

    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "diagnose", "check", pile_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Pile appears healthy"))
        .stdout(predicate::str::contains(
            "Recognized 1 inert legacy V3 collection record(s) (first byte 0, last byte 0)",
        ));
}

#[test]
fn diagnose_record_at_distinguishes_version_skew_from_a_torn_record() {
    let dir = tempdir().unwrap();
    let unsupported_path = dir.path().join("unsupported.pile");
    let torn_path = dir.path().join("torn.pile");

    let mut unsupported = vec![0u8; 256];
    unsupported[..16].fill(0xA5);
    std::fs::write(&unsupported_path, unsupported).unwrap();
    std::fs::write(
        &torn_path,
        hex::decode("9C33EEB525065A62EAEC4BE43DCC355A").expect("V3 blob marker"),
    )
    .unwrap();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "diagnose",
            "record-at",
            unsupported_path.to_str().unwrap(),
            "0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "record format unsupported by this binary",
        ))
        .stderr(predicate::str::contains("Upgrade trible"))
        .stderr(predicate::str::contains("amputate").not());

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "diagnose",
            "record-at",
            torn_path.to_str().unwrap(),
            "0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("malformed or incomplete"))
        .stderr(predicate::str::contains("cannot prove"))
        .stderr(predicate::str::contains("--truncate-to <BYTE_OFFSET>"));
}

#[test]
fn diagnose_qualifies_health_when_opaque_bodies_are_skipped() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("opaque-diag.pile");
    std::fs::write(&pile_path, opaque_envelope(None)).unwrap();

    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "diagnose", "check", pile_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Known record projection appears healthy",
        ))
        .stdout(predicate::str::contains(
            "bodies were not semantically validated",
        ));
}

#[test]
fn diagnose_locate_hash_scans_the_complete_opaque_record() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("opaque-locate.pile");
    let needle = [0x4D; 32];
    std::fs::write(&pile_path, opaque_envelope(Some(needle))).unwrap();
    let handle = format!("blake3:{}", hex::encode_upper(needle));

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "diagnose",
            "locate-hash",
            pile_path.to_str().unwrap(),
            &handle,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("opaque-record byte match"))
        .stdout(predicate::str::contains("opaque records: 1"));
}

#[test]
fn diagnose_reports_invalid_hash() {
    use std::io::Seek;
    use std::io::Write;

    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("bad.pile");
    std::fs::File::create(&pile_path).unwrap();
    let blob_path = dir.path().join("blob.bin");
    std::fs::write(&blob_path, b"good data").unwrap();

    // put a blob into the pile
    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "put",
            pile_path.to_str().unwrap(),
            blob_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // corrupt the blob bytes directly
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(&pile_path)
        .unwrap();
    // first blob payload starts after the fixed 256-byte envelope header
    file.seek(std::io::SeekFrom::Start(256)).unwrap();
    file.write_all(b"X").unwrap();

    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "diagnose", "check", pile_path.to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains("incorrect hashes"));
}

#[test]
fn inspect_outputs_tribles() {
    use triblespace::prelude::*;
    use triblespace_core::examples;
    use triblespace_core::inline::encodings::hash::Handle;

    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("inspect.pile");
    std::fs::File::create(&pile_path).unwrap();

    use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
    use triblespace_core::blob::{Blob, IntoBlob};
    let dataset = examples::dataset();
    let blob: Blob<SimpleArchive> = dataset.to_blob();

    let handle_str = {
        let mut pile: Pile = Pile::open(&pile_path).unwrap();
        let handle = pile.put::<SimpleArchive, _>(blob).unwrap();
        pile.close().unwrap();

        let hash = Handle::to_hash(handle);
        hash.from_inline::<String>()
    };

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "inspect",
            pile_path.to_str().unwrap(),
            &handle_str,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Length:"));
}

#[test]
fn diagnose_locate_hash_reports_header_and_payload_refs() {
    let dir = tempdir().unwrap();
    let pile_path = dir.path().join("locate_hash.pile");
    std::fs::File::create(&pile_path).unwrap();

    // Put blob1 and capture its handle string.
    let blob1_path = dir.path().join("blob1.bin");
    std::fs::write(&blob1_path, b"blob1").unwrap();
    let out1 = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "put",
            pile_path.to_str().unwrap(),
            blob1_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let handle1 = String::from_utf8(out1).unwrap();
    let handle1 = handle1.trim().to_string();

    // Put blob2 containing the raw digest bytes of blob1 in its payload, so the
    // locator can find a payload reference.
    let digest_hex = handle1.strip_prefix("blake3:").expect("handle prefix");
    let digest_bytes = hex::decode(digest_hex).expect("decode digest hex");
    let mut payload = b"prefix".to_vec();
    payload.extend_from_slice(&digest_bytes);
    payload.extend_from_slice(b"suffix");

    let blob2_path = dir.path().join("blob2.bin");
    std::fs::write(&blob2_path, payload).unwrap();
    let out2 = Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "blob",
            "put",
            pile_path.to_str().unwrap(),
            blob2_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let handle2 = String::from_utf8(out2).unwrap();
    let handle2 = handle2.trim().to_string();

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "diagnose",
            "locate-hash",
            pile_path.to_str().unwrap(),
            &handle1,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("blob header match"))
        .stdout(predicate::str::contains(&format!(
            "payload reference in {handle2}"
        )))
        .stdout(predicate::str::contains("Summary"));
}

/// A malformed or incomplete source pile must make `migrate` fail loud with a
/// boundary-confirmed repair path, without truncating the source file.
#[test]
fn corrupt_source_fails_loud_without_truncation() {
    use std::io::Write;

    let dir = tempdir().unwrap();
    let src_path = dir.path().join("corrupt_src.pile");
    std::fs::File::create(&src_path).unwrap();

    // Tear the tail before even a complete record marker has landed.
    {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&src_path)
            .unwrap();
        file.write_all(&[0xFFu8; 8]).unwrap();
        file.sync_all().unwrap();
    }
    let len_before = std::fs::metadata(&src_path).unwrap().len();

    let fail_loud = || {
        predicate::str::contains("cannot prove")
            .and(predicate::str::contains("--truncate-to <BYTE_OFFSET>"))
    };

    // migrate (in-place rewrite): still refuses to open a corrupt pile.
    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "migrate", src_path.to_str().unwrap(), "list"])
        .assert()
        .failure()
        .stderr(fail_loud());

    let len_after = std::fs::metadata(&src_path).unwrap().len();
    assert_eq!(
        len_before, len_after,
        "source pile must not be truncated by a failed open"
    );
}

/// A complete unknown marker is format/version skew, not evidence that the
/// tail is disposable. Normal commands and the explicit repair command both
/// fail without suggesting or performing truncation.
#[test]
fn unsupported_record_marker_never_recommends_or_performs_amputation() {
    use std::io::Write;

    let dir = tempdir().unwrap();
    let path = dir.path().join("unsupported.pile");
    let unknown_marker = [0xA5u8; 16];
    let mut unknown_record = [0u8; 256];
    unknown_record[..16].copy_from_slice(&unknown_marker);
    std::fs::File::create(&path)
        .unwrap()
        .write_all(&unknown_record)
        .unwrap();
    let len_before = std::fs::metadata(&path).unwrap().len();

    let unsupported_without_repair_hint = || {
        predicate::str::contains("unsupported")
            .and(predicate::str::contains("version skew"))
            .and(predicate::str::contains("trible pile amputate").not())
    };

    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "migrate", path.to_str().unwrap(), "list"])
        .assert()
        .failure()
        .stderr(unsupported_without_repair_hint());

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "amputate",
            path.to_str().unwrap(),
            "--truncate-to",
            "0",
        ])
        .assert()
        .failure()
        .stderr(unsupported_without_repair_hint());

    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        len_before,
        "an unknown marker must survive even an explicit amputation attempt"
    );
}

#[test]
fn amputate_requires_and_matches_the_current_reader_boundary() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("torn.pile");
    std::fs::write(&path, [0xFFu8; 8]).unwrap();

    // The old copy-pasteable command is deliberately incomplete now.
    Command::cargo_bin("trible")
        .unwrap()
        .args(["pile", "amputate", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--truncate-to"));
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 8);

    // A guessed boundary cannot destroy anything.
    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "amputate",
            path.to_str().unwrap(),
            "--truncate-to",
            "1",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "does not match the current reader's boundary 0",
        ));
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 8);

    Command::cargo_bin("trible")
        .unwrap()
        .args([
            "pile",
            "amputate",
            path.to_str().unwrap(),
            "--truncate-to",
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("at confirmed boundary"));
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
}
