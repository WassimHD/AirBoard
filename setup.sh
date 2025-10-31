if [ ! -d ".venv" ]; then
    echo 'creating virtual environment...   '
    python3 -m venv venv
else 
    echo 'virtual environment already exists.   '
fi

echo 'activating virtual environment...   '
source .venv/bin/activate


#check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo 'installing dependencies...   '
    pip3 install -r requirements.txt
    echo 'setup complete!   '
else
    echo 'requirements.txt not found. Please create one and run the script again.   '
fi
