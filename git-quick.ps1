# git-quick.ps1

git add .

if ($args.Count -eq 0) {
    $commit_message = "Automatic commit"
} else {
    $commit_message = $args[0]
}

git commit -m "$commit_message"

if ($LASTEXITCODE -eq 0) {
    Write-Output "Commit successful. Pushing to main branch..."
    git push origin main
    Write-Output "Push completed."
} else {
    Write-Output "Commit failed. Please check for errors."
}
