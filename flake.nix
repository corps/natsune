{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:

      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            uv
            pkgs.python314
            pkgs.pre-commit
          ];

          shellHook = ''
            source .env || true
            export XDG_CONFIG_HOME="$HOME/.config"
            export REPO_ROOT=$(pwd)

            uv sync --all-extras
            unset PYTHONPATH
            source .venv/bin/activate

            pre-commit install -t pre-commit || true
          '';
        };

        checks = { };
      }
    );
}
