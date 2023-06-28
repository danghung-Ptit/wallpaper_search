#!/bin/bash
echo "start gunicorn server"
gunicorn --workers 1 --name app -b 0.0.0.0:8000 --reload app.run_web_server:app
