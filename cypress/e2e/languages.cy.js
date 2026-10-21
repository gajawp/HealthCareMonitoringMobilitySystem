describe("Chatbot language selection", () => {
  const chatPanel = ".st-key-healthbridge_chat_panel";

  const languageCombobox =
    '[data-testid="stSelectbox"] input[role="combobox"]';

  beforeEach(() => {
    cy.env(["TEST_USERNAME", "TEST_PASSWORD"]).then(
      ({ TEST_USERNAME, TEST_PASSWORD }) => {
        cy.login(TEST_USERNAME, TEST_PASSWORD);
      }
    );

    cy.contains("button", /^Chat/i)
      .should("be.visible")
      .click();

    cy.get(chatPanel, {
      timeout: 20000,
    }).should("be.visible");
  });

  function getChatInput() {
    return cy
      .get(chatPanel)
      .find('[data-testid="stTextInput"] input')
      .filter(":visible")
      .first();
  }

  function openLanguageSettings() {
    cy.get(chatPanel).then(($panel) => {
      const selectbox = $panel.find(
        '[data-testid="stSelectbox"]'
      );

      if (!selectbox.is(":visible")) {
        cy.wrap($panel)
          .contains("🌐 Language and settings")
          .should("be.visible")
          .click();
      }
    });

    cy.get(chatPanel)
      .find('[data-testid="stSelectbox"]')
      .should("be.visible");
  }

  function selectLanguage(language) {
    openLanguageSettings();

    /*
     * Streamlit renders the arrow dynamically without a stable
     * selector. Find the element located at the right side of
     * the combobox and click it.
     */
    cy.get(chatPanel)
      .find(languageCombobox)
      .should("be.visible")
      .then(($input) => {
        const input = $input[0];
        const rectangle =
          input.getBoundingClientRect();
        const document = input.ownerDocument;

        const arrow =
          document.elementFromPoint(
            rectangle.right - 18,
            rectangle.top +
              rectangle.height / 2
          );

        expect(
          arrow,
          "language dropdown arrow"
        ).not.to.be.null;

        cy.wrap(arrow).click({
          force: true,
        });
      });

    // Verify that the dropdown opened.
    cy.get(chatPanel)
      .find(languageCombobox)
      .should(
        "have.attr",
        "aria-expanded",
        "true"
      );

    /*
     * Select the visible option with a native click.
     * This avoids Cypress detached-element errors when
     * Streamlit immediately rerenders.
     */
    cy.contains(language, {
      timeout: 10000,
    })
      .filter(":visible")
      .then(($options) => {
        expect(
          $options.length
        ).to.be.greaterThan(0);

        const option =
          $options[$options.length - 1];

        option.click();
      });

    // Confirm that Streamlit retained the selection.
    cy.get(chatPanel, {
      timeout: 20000,
    })
      .find(languageCombobox)
      .should("have.value", language);
  }

  function submitQuestion(question) {
    cy.get(chatPanel)
      .find('[data-testid="stChatMessage"]')
      .its("length")
      .then((initialMessageCount) => {
        getChatInput()
          .should("be.visible")
          .and("be.enabled")
          .clear()
          .type(question)
          .should(
            "have.value",
            question
          );

        // Click the arrow button to send.
        cy.get(chatPanel)
          .contains("button", /^➤$/)
          .should("be.visible")
          .and("be.enabled")
          .click();

        // Verify that submission cleared the input.
        getChatInput()
          .should("be.visible")
          .and("have.value", "");

        // Verify that new chat messages were added.
        cy.get(chatPanel, {
          timeout: 60000,
        })
          .find(
            '[data-testid="stChatMessage"]'
          )
          .should(($messages) => {
            expect(
              $messages.length
            ).to.be.greaterThan(
              initialMessageCount
            );
          });
      });
  }

  it("uses English by default", () => {
    getChatInput()
      .should("be.visible")
      .and(
        "have.attr",
        "placeholder",
        "Ask about sessions, alerts, mobility metrics, or exercises"
      );
  });

  it("changes the interface to Spanish", () => {
    selectLanguage("Spanish - Español");

    getChatInput()
      .should("be.visible")
      .and(
        "have.attr",
        "placeholder",
        "Pregunte sobre sesiones, alertas, métricas de movilidad o ejercicios"
      );
  });

  it("changes the interface to German", () => {
    selectLanguage("German - Deutsch");

    getChatInput()
      .should("be.visible")
      .and(
        "have.attr",
        "placeholder",
        "Fragen Sie nach Sitzungen, Warnungen, Mobilitätswerten oder Übungen"
      );
  });

  // it("changes the interface to Telugu", () => {
  //   selectLanguage("Telugu - తెలుగు");

  //   getChatInput()
  //     .should("be.visible")
  //     .and(
  //       "have.attr",
  //       "placeholder",
  //       "సెషన్లు, హెచ్చరికలు, మొబిలిటీ కొలతలు లేదా వ్యాయామాల గురించి అడగండి"
  //     );
  // });

});