use std::process::Command;
pub fn main() {
    // Skip this build script when building documentation on docs.rs
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    // Install setuptools
    let output = Command::new("pip")
        .arg("install")
        .arg("setuptools")
        .output()
        .expect("Failed to execute command");
    if !output.status.success() {
        eprintln!("Command failed with status: {}", output.status);
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Installation of setuptools failed");
    }

    // Run distutils to get the library path
    let python_inline_script =
        "from distutils import sysconfig;print(sysconfig.get_config_var('LIBDIR'))";
    let output = Command::new("python")
        .arg("-c")
        .arg(python_inline_script)
        .output()
        .expect("Failed to get LIBDIR");
    if !output.status.success() {
        eprintln!("Python command failed with status: {}", output.status);
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Python command failed");
    }
    let python_lib_path = String::from_utf8_lossy(&output.stdout);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", python_lib_path);
}
