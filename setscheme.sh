#! /bin/bash
# 1: image path
# 2: colourscheme name
# First, check if the image file exists. If not, exit
if [[ ! -a "$1" ]]; then
	echo "Could not locate file $1"
	exit 1
fi

# Check if the colourscheme file is valid
if [[ -z "$2" ]]; then
	echo "Invalid colourscheme name given"
	exit 1
fi

# If no colouscheme directory exists, create it
if [[ ! -d "$HOME/.config/colours/colourschemes" ]]; then
	mkdir $HOME/.config/colours/colourschemes
	echo "Made colourscheme directory"
fi

# Then check if the colourscheme file exists. If not, generate it

if [[ ! -a "$HOME/.config/colours/colourschemes/$2" ]]; then
	echo "Generating colourscheme"
	python $HOME/.config/colours/colourscheme.py "$1" -s 10 > "$HOME/.config/colours/colourschemes/$2"
fi

# Then do a symbolic link from that colourscheme to .config/colours/current
if [[ -L "$HOME/.config/colours/current" ]]; then
	rm "$HOME/.config/colours/current"
	echo "Removed existing colourscheme symlink"
fi

ln -s "$HOME/.config/colours/colourschemes/$2" "$HOME/.config/colours/current"

# xrdb -merge

xrdb -merge "$HOME/.Xresources"

# Run the update script

python "$HOME/.config/colours/colours.py"

# Set the desktop background

if [[ -L "$HOME/.config/colours/currentbg" ]]; then
	rm "$HOME/.config/colours/currentbg"
	echo "Removed existing background symlink"
fi

ln -s "$1" "$HOME/.config/colours/currentbg"
feh --bg-fill "$(readlink "$HOME/.config/colours/currentbg")"
