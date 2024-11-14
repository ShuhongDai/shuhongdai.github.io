#!/bin/bash


# Stage all changes
git add .

# Check if a commit message was provided as an argument
if [ -z "$1" ]; then
  # If no message provided, use a default commit message
  commit_message="Automatic commit"
else
  # Use the provided message
  commit_message="$1"
fi

# Commit changes with the determined message
git commit -m "$commit_message"

# Check if commit was successful
if [ $? -eq 0 ]; then
  echo "Commit successful. Pushing to main branch..."
  # Push changes to the main branch
  git push origin main
  echo "Push completed."
else
  echo "Commit failed. Please check for errors."
fi