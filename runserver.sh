#!/bin/bash
echo "start uvicorn server"
exec uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
