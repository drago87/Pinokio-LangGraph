module.exports = {
  run: [{
    method: "fs.rm",
    params: {
      path: "venv"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "data"
    }
  }]
}
