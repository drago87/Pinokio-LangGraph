module.exports = {
  run: [{
    method: "shell.run",
    params: {
      "message": "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      "message": [
        "uv venv venv",
        "uv pip install -r requirements.txt --upgrade"
      ]
    }
  }, {
    method: "input",
    "params": {
      "title": "Update Complete",
      "description": "Dependencies updated. Click Start to restart the server."
    }
  }]
}
