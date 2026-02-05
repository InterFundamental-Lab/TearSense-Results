import os
import sys
import tokenize
import argparse
from io import BytesIO

def remove_comments_from_source(source_code):
    """
    Parses python source code and removes comments using the tokenizer.
    Preserves strings that contain '#' characters.
    """
    io_obj = BytesIO(source_code.encode('utf-8'))
    out = ""
    last_lineno = -1
    last_col = 0
    
    try:
        tokens = tokenize.tokenize(io_obj.readline)
        
        # We collect valid tokens to untokenize later
        result_tokens = []
        
        for tok in tokens:
            token_type = tok.type
            token_string = tok.string
            
            # Skip Comment Tokens
            if token_type == tokenize.COMMENT:
                continue
            
            # Skip Docstrings (Optional - currently set to KEEP docstrings)
            # If you want to remove docstrings too, uncomment the logic below:
            # if token_type == tokenize.STRING:
            #    pass # Logic to detect if it's a docstring vs variable assignment is complex
            
            result_tokens.append((token_type, token_string))
            
        # Untokenize reconstructs the code from the filtered tokens
        out = tokenize.untokenize(result_tokens).decode('utf-8')
        return out

    except tokenize.TokenError:
        # If there's a syntax error in the file, return original
        return source_code

def process_file(file_path, force=False):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check if file even has comments to save IO
        if '#' not in source:
            return False

        clean_source = remove_comments_from_source(source)
        
        if clean_source.strip() == source.strip():
            return False

        if force:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(clean_source)
            print(f"[CLEANED] {file_path}")
        else:
            print(f"[DRY RUN] Would clean: {file_path}")
        
        return True

    except Exception as e:
        print(f"[ERROR] Could not process {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Recursively remove Python comments.")
    parser.add_argument("directory", help="Target directory to clean")
    parser.add_argument("--force", action="store_true", help="Actually modify files (default is Dry Run)")
    parser.add_argument("--ignore", nargs="+", default=["venv", ".git", "__pycache__", "env"], help="Folders to ignore")
    
    args = parser.parse_args()
    
    target_dir = args.directory
    
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"RUNNING COMMENT REMOVER ({'DESTRUCTIVE MODE' if args.force else 'DRY RUN'})")
    print(f"Target: {os.path.abspath(target_dir)}")
    print(f"{'='*60}")

    count = 0
    for root, dirs, files in os.walk(target_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in args.ignore]
        
        for file in files:
            if file.endswith(".py") and file != "comment_remover.py":
                path = os.path.join(root, file)
                if process_file(path, force=args.force):
                    count += 1

    print(f"{'='*60}")
    if not args.force:
        print(f"Dry run complete. Found {count} files to clean.")
        print("Run again with --force to apply changes.")
    else:
        print(f"Done. Cleaned {count} files.")

if __name__ == "__main__":
    main()