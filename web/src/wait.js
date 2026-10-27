// Feedback while a live call runs: the models take a few seconds per request.

const SPINNER = '<span class="spinner" aria-hidden="true"></span>';

// Put a button into a busy state; returns a function that restores it
export function busy(button, label) {
  const html = button.innerHTML;
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.innerHTML = `${SPINNER}${label}`;
  return () => {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    button.innerHTML = html;
  };
}

// A hint line with a spinner, for an element that otherwise holds plain text
export function waiting(el, text) {
  el.innerHTML = `${SPINNER}${text}`;
  el.classList.add("waiting");
  return () => el.classList.remove("waiting");
}
