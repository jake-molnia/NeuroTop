{
  description = "dev-env";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.just
            pkgs.uv
            pkgs.python312 # For comparing with uv
            (pkgs.python312.withPackages (ps: [ ps.tkinter ]))
          ];

          shellHook = ''
            echo "Ready!"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"
            echo "just: $(just --version)"
          '';
        };
      }
    );
}
