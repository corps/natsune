codegen:
	nix develop -c python -c "import natsune.control_flow; from karakuri.codegen_buffer import write_generated_types; write_generated_types()"
