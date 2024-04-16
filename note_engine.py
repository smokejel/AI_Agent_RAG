from llama_index.core.tools import FunctionTool
import os

# Define the path of notes text file
note_file = os.path.join("data", "notes.txt")

def save_note(note):
    # If the file does not exist, create the file and open it with write access
    if not os.path.exists(note_file):
        open(note_file,"w")

    # If the does exist, open the file in append mode and add lines to the file with a new line at the end
    with open(note_file, "a") as f:
        f.writelines([note + "\n"])
        f.close()

    return "note saved"

# Now we wrap this function into tool that the engine can leverage. Function referenced the save_note object.
# Name and description should be well-defined for the llm to know how to use the tool.
note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="This tool can save a text based note to a file for the user"
)
