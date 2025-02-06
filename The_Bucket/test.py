text = input("Enter up to 255 characters: ")[:255]
pattern = input("Enter the pattern to replace: ")
replacement = input("Enter the replacement: ")
print(text.replace(pattern, replacement))