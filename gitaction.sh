tfile=$(mktemp /tmp/config.XXXXXXXXX)
GITCONF=".git/config"
commitmsg=${1:-git repository initialised}
if [ -f "$GITCONF" ]; then
mv "$GITCONF" tfile
rm -rf .git
git init .
mv tfile "$GITCONF"
git add .
git commit -a -m "${commitmsg}"
git push -f
else
echo "Warning: No git config file found. Aborting.";exit;
fi