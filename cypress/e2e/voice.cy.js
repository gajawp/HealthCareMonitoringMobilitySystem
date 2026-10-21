describe("Chatbot voice control", () => {
  const chatPanel = ".st-key-healthbridge_chat_panel";

  beforeEach(() => {
    cy.env(["TEST_USERNAME", "TEST_PASSWORD"]).then(
      ({ TEST_USERNAME, TEST_PASSWORD }) => {
        cy.login(TEST_USERNAME, TEST_PASSWORD);
      }
    );

    cy.contains("button", /^Chat/i).click();

    cy.get(chatPanel, { timeout: 20000 })
      .should("be.visible");
  });

  // st.audio_input renders natively (no iframe) with a text label.
  function getMicrophone() {
    return cy
      .get(chatPanel)
      .find('[data-testid="stAudioInput"]')
      .filter(":visible")
      .first()
      .should("be.visible");
  }

  it("displays the microphone component", () => {
    getMicrophone().should("contain.text", "Record a spoken question");
  });

  it("displays an enabled microphone button", () => {
    getMicrophone()
      .find("button")
      .filter(":visible")
      .first()
      .should("be.visible")
      .and("be.enabled");
  });

  it("allows the microphone button to be clicked", () => {
    getMicrophone()
      .find("button")
      .filter(":visible")
      .first()
      .should("be.enabled")
      .click();

    getMicrophone()
      .find("button")
      .filter(":visible")
      .first()
      .should("be.visible");
  });

  it("keeps the text field available with voice enabled", () => {
    cy.get(chatPanel)
      .find('[data-testid="stTextInput"] input')
      .filter(":visible")
      .first()
      .should("be.visible")
      .and("be.enabled");
  });

  it("displays voice control after changing language", () => {
    cy.get(chatPanel)
      .contains("🌐 Language and settings")
      .click();

    cy.get(chatPanel)
      .find('[data-testid="stSelectbox"]')
      .find('[data-baseweb="select"]')
      .click();

    cy.get('[role="option"]')
      .contains("Spanish - Español")
      .click();

    cy.get(chatPanel, { timeout: 20000 })
      .should("be.visible");

    getMicrophone()
      .find("button")
      .filter(":visible")
      .first()
      .should("be.visible")
      .and("be.enabled");
  });
});