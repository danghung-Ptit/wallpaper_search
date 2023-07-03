#!/bin/bash
echo "start gunicorn server"
gunicorn --workers 4 --name app -b 0.0.0.0:3000 --reload app.main:app
