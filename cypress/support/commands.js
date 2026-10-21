Cypress.Commands.add("login", (username, password) => {
  cy.visit("/");

  cy.get('[data-testid="stTextInput"]')
    .should("have.length.at.least", 2);

  cy.get('[data-testid="stTextInput"]')
    .eq(0)
    .find("input")
    .should("be.visible")
    .clear()
    .type(username)
    .should("have.value", username);

  cy.get('[data-testid="stTextInput"]')
    .eq(1)
    .find("input")
    .should("be.visible")
    .clear()
    .type(password, { log: false })
    .should("have.value", password);

  cy.contains("button", /^Login$/)
    .should("be.enabled")
    .click();

  cy.contains("button", /^Logout$/, {
    timeout: 20000,
  }).should("be.visible");
});
Cypress.Commands.add("getChatBody", () => {
  return cy
    .get("iframe", { timeout: 20000 })
    .filter(":visible")
    .last()
    .should("be.visible")
    .its("0.contentDocument.body")
    .should("not.be.empty")
    .then((body) => cy.wrap(body));
});