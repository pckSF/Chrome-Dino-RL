// Javascript to simulate holding down a key by using the following:
//
// press the key "a"
// browser.executeScript(SIMULATE_KEY, "keydown", "a", target);
//
// release the key "a"
// browser.executeScript(SIMULATE_KEY, "keyup", "a", target);

const SIMULATE_KEY =
  "var e = new Event(arguments[0]);" +
  "e.key = arguments[1];" +
  "e.keyCode = e.key.charCodeAt(0);" +
  "e.which = e.keyCode;" +
  "e.altKey = false;" +
  "e.ctrlKey = false;" +
  "e.shiftKey = false;" +
  "e.metaKey = false;" +
  "e.bubbles = true;" +
  "arguments[2].dispatchEvent(e);";