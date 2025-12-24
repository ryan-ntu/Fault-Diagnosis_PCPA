git ls-files
git rm --cached *.pt

git branch -d branch_name
git branch -D branch_name


git add .gitignore
git commit -m "Update .gitignore"

git rm -r --cached .
git add -A
git commit -m "Apply .gitignore and stop tracking ignored files"