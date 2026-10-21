describe("Healthcare chatbot", () => {
  const chatPanel = ".st-key-healthbridge_chat_panel";

  beforeEach(() => {
    cy.env(["TEST_USERNAME", "TEST_PASSWORD"]).then(
      ({ TEST_USERNAME, TEST_PASSWORD }) => {
        cy.login(TEST_USERNAME, TEST_PASSWORD);
      }
    );
  });

  function openChatbot() {
    cy.contains("button", /^Chat/i)
      .should("be.visible")
      .click();

    cy.get(chatPanel, { timeout: 20000 })
      .should("be.visible");
  }

  function getChatInput() {
    return cy
      .get(chatPanel)
      .find('[data-testid="stTextInput"] input')
      .filter(":visible")
      .first();
  }

  it("displays the chatbot launcher", () => {
    cy.contains("button", /^Chat/i)
      .should("be.visible")
      .and("be.enabled");
  });

  it("opens the chatbot and displays the text box", () => {
    openChatbot();

    getChatInput()
      .should("be.visible")
      .and(
        "have.attr",
        "placeholder",
        "Ask about sessions, alerts, mobility metrics, or exercises"
      );
  });

  it("enters and submits a mobility question", () => {
    openChatbot();

    const question = "How many repetitions did I complete?";

    getChatInput()
      .should("be.visible")
      .clear()
      .type(question)
      .should("have.value", question);

    // Click the actual Send button.
    cy.get(chatPanel)
        .contains("button", /^➤$/)
        .should("be.visible")
        .and("be.enabled")
        .click();

    // Confirm the submitted question appears in the conversation.
    cy.get(chatPanel, { timeout: 30000 })
      .contains(question)
      .should("be.visible");

    // Confirm that the assistant produces an answer.
    cy.get(chatPanel, { timeout: 60000 })
      .contains(/repetition|repetitions|reps/i)
      .should("be.visible");
  });

  it("displays the voice control", () => {
    openChatbot();

    // st.audio_input renders natively with a descriptive label.
    cy.get(chatPanel)
      .find('[data-testid="stAudioInput"]')
      .filter(":visible")
      .first()
      .should("be.visible")
      .find("button")
      .filter(":visible")
      .first()
      .should("be.visible");
  });

  it("allows logout", () => {
    cy.contains("button", /^Logout$/)
      .should("be.visible")
      .click();

    cy.contains("button", /^Login$/, {
      timeout: 15000,
    }).should("be.visible");
  });
});