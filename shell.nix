with import <nixpkgs> {};

stdenv.mkDerivation {
  name = "rust-env";
  nativeBuildInputs = [
    rustup 
    cmake
    pkg-config
    freetype
    expat
  ];

  
  # Set Environment Variables
  RUST_BACKTRACE = 1;
}
