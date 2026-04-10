module.exports = {
  run: [{
    when: "{{platform === 'win32'}}",
    method: "shell.run",
    params: {
      "message": "rmdir /s /q venv config.ini settings.json dbs 2>nul & echo Reset complete"
    }
  }, {
    when: "{{platform !== 'win32'}}",
    method: "shell.run",
    params: {
      "message": "rm -rf venv config.ini settings.json dbs/"
    }
  }, {
    method: "input",
    params: {
      "title": "Reset Complete",
      "description": "PinokioLangGraph has been reset to pre-install state.\n\nClick Install to set it up again."
    }
  }]
}