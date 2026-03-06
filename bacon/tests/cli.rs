use std::process::Command;

#[test]
fn help_smoke_test() {
    let output = Command::new(env!("CARGO_BIN_EXE_bacon"))
        .arg("--help")
        .output()
        .unwrap();

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage:"));
    assert!(stdout.contains("--cpu"));
    assert!(stdout.contains("--count"));
}

#[test]
fn version_smoke_test() {
    let output = Command::new(env!("CARGO_BIN_EXE_bacon"))
        .arg("--version")
        .output()
        .unwrap();

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("bacon"));
    assert!(stdout.contains("0.1.0"));
}
