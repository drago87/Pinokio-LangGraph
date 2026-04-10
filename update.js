module.exports = {
  run: [{
    method: "shell.run",
    params: {
      "message": "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      "message": "uv pip install -r requirements.txt"
    }
  }, {
    method: "input",
    params: {
      "title": "Update Complete",
      "description": "Agent-StateSync updated. Click Start to restart the server."
    }
  }]
}