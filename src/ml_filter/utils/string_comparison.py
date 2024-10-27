import difflib

def get_char_differences(str1: str, str2: str) -> int:
    diff = difflib.ndiff(str1, str2)
    differences = [(i, d[0], d[-1]) for i, d in enumerate(diff) if d[0] in ('-', '+')]
    
    # Collect differences in one line
    diff_output = ", ".join([f"{change_type}{char}" for index, change_type, char in differences])
    
    # Print the differences in a single line
    print("Differences:", diff_output)
    
    # Return the count of differences
    return len(differences)