describe("Login interface", () => {
  beforeEach(() => {
    cy.visit("/");
  });

  it("displays the login interface", () => {
    cy.get('[data-testid="stSelectbox"]')
      .should("be.visible");

    cy.get('[data-testid="stTextInput"]')
      .should("have.length.at.least", 2);

    cy.get('[data-testid="stTextInput"]')
      .eq(0)
      .find("input")
      .should("be.visible");

    cy.get('[data-testid="stTextInput"]')
      .eq(1)
      .find("input")
      .should("be.visible");

    cy.contains("button", /^Login$/)
      .should("be.visible")
      .and("be.enabled");

    cy.contains(/Configured demo accounts/i)
      .should("be.visible");
  });

  it("shows an error for invalid credentials", () => {
    cy.get('[data-testid="stTextInput"]')
      .eq(0)
      .find("input")
      .clear()
      .type("INVALID_USER");

    cy.get('[data-testid="stTextInput"]')
      .eq(1)
      .find("input")
      .clear()
      .type("wrong-password", { log: false });

    cy.contains("button", /^Login$/).click();

    cy.contains(/Invalid username or password/i, {
      timeout: 15000,
    }).should("be.visible");
  });

  it("logs in with valid patient credentials", () => {
    cy.env(["TEST_USERNAME", "TEST_PASSWORD"]).then(
      ({ TEST_USERNAME, TEST_PASSWORD }) => {
        cy.login(TEST_USERNAME, TEST_PASSWORD);
      }
    );

    cy.contains(/Welcome, Demo Patient!/i, {
      timeout: 20000,
    }).should("be.visible");

    cy.contains(/Patient ID:\s*P1001/i)
      .should("be.visible");

    cy.contains("button", /^Logout$/)
      .should("be.visible");

    cy.contains("button", /Chat/i)
      .should("be.visible");
  });
});