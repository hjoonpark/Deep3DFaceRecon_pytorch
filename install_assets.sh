#!/bin/sh
ver="`cat VERSION`"
echo $ver
url="https://pbss.s8k.io/v1/AUTH_team-digital-human/image2avatar/artifactory/AvatarGeneratorCoreAssets/$ver.zip?temp_url_sig=901d6eca6ef325708c5a2866b3a91190f1ce0605&temp_url_expires=2044-01-01T02:55:34Z&temp_url_prefix=artifactory/AvatarGeneratorCoreAssets/&inline"
echo "curling"
curl -fsSL "$url" -L -o assets.zip --progress-bar
echo "done"
unzip assets.zip
# rm -rf assets.zip
