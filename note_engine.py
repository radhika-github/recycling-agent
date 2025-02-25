import os
from llama_index.core.tools import FunctionTool

def save_note_from_file(file_path):
    """Save content from a provided text file to storage."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    with open(file_path, "r") as f:
        content = f.read()

    if not content.strip():
        return "The provided file is empty."

    save_note(content, "data/notes.txt")  # Default save location
    return f"Content from {file_path} saved successfully."

def save_note(note, file_path="data/notes.txt"):
    """Save a note to a specified file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode) as f:
        content = f"\n" + note if mode == "a" else note
        f.write(content)

    return f"Note saved to {file_path}."

def read_notes(file_path="data/notes.txt"):
    """Read notes from a specified file."""
    if not os.path.exists(file_path):
        return "No notes found."

    with open(file_path, "r") as f:
        return f.read()

def create_note_tools(default_file="data/notes.txt"):
    """Create FunctionTools for saving, reading, and importing notes from a file."""
    save_tool = FunctionTool.from_defaults(
        fn=lambda note: save_note(note, default_file),
        name="note_saver",
        description="This tool saves text-based notes to a file."
    )

    read_tool = FunctionTool.from_defaults(
        fn=lambda: read_notes(default_file),
        name="note_reader",
        description="This tool reads saved notes from a file."
    )

    import_tool = FunctionTool.from_defaults(
        fn=save_note_from_file,
        name="note_importer",
        description="This tool imports notes from an uploaded text file."
    )

    return save_tool, read_tool, import_tool

# Example usage
save_tool, read_tool, import_tool = create_note_tools("data/my_notes.txt")

# Save a note manually
print(save_tool.fn("This is a new note."))

# Read saved notes
print(read_tool.fn())

# Import notes from another text file
print(import_tool.fn("data/notes_to_import.txt"))
