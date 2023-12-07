TARGET_DIR=$(pwd)/games

echo 'Installing StarCraft II in the working directory...'

# Check if the StarCraftII directory exists
if [ ! -d $TARGET_DIR ]; then
    echo 'StarCraftII is not installed. Installing now...'
    # Create the StarCraftII directory
    mkdir -p $TARGET_DIR
    # Download the StarCraftII package
    wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    # Unzip the package into the StarCraftII directory
    unzip -P iagreetotheeula SC2.4.10.zip -d $TARGET_DIR
    # Remove the downloaded zip file to clean up
    rm SC2.4.10.zip
else
    echo 'StarCraftII is already installed.'
fi

echo 'StarCraft II is installed in: '$TARGET_DIR

# Installing SMAC Maps
echo 'Installing SMAC maps...'

MAP_DIR="$TARGET_DIR/StarCraftII/Maps"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
    mkdir -p $MAP_DIR
fi

# Download and unzip SMAC Maps into the Maps directory
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip -d $MAP_DIR
rm SMAC_Maps.zip

echo 'StarCraft II and all maps are installed.'
