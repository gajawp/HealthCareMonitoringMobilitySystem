export function getStreamlitFrameBody(expectedText) {
  return cy
    .get("iframe", {
      timeout: 120000
    })
    .should(($iframes) => {
      const matchingFrame = [...$iframes].find((iframe) => {
        try {
          const text =
            iframe.contentDocument?.body?.innerText || "";

          return text.includes(expectedText);
        } catch (error) {
          return false;
        }
      });

      expect(
        matchingFrame,
        `iframe containing "${expectedText}"`
      ).to.exist;
    })
    .then(($iframes) => {
      const matchingFrame = [...$iframes].find((iframe) => {
        try {
          const text =
            iframe.contentDocument?.body?.innerText || "";

          return text.includes(expectedText);
        } catch (error) {
          return false;
        }
      });

      return cy.wrap(
        matchingFrame.contentDocument.body,
        { log: false }
      );
    });
}

export function loginAsDemoPatient() {
  cy.visit("/", {
    timeout: 120000,
    failOnStatusCode: false
  });

  getStreamlitFrameBody("User ID").within(() => {
    cy.get('[data-testid="stTextInput"]')
      .eq(0)
      .find("input")
      .should("be.visible")
      .clear()
      .type("P1001");

    cy.get('[data-testid="stTextInput"]')
      .eq(1)
      .find("input")
      .should("be.visible")
      .clear()
      .type("pass123", {
        log: false
      });

    cy.get('[data-testid="stTextInput"]')
      .eq(0)
      .find("input")
      .should("have.value", "P1001");

    cy.get('[data-testid="stTextInput"]')
      .eq(1)
      .find("input")
      .should("have.value", "pass123");

    cy.contains("button", /^Login$/)
      .should("be.visible")
      .and("be.enabled")
      .click();
  });

  getStreamlitFrameBody("Demo Patient").within(() => {
    cy.contains("Demo Patient", {
      timeout: 120000
    }).should("be.visible");

    cy.contains("Patient ID: P1001")
      .should("be.visible");
  });
}

export function getChatInput() {
  return getStreamlitFrameBody("Suggested questions")
    .find(
      'input[placeholder*="Ask about"], ' +
      'textarea[placeholder*="Ask about"]'
    )
    .filter(":visible")
    .first();
}

export function openChatbot() {
  getStreamlitFrameBody("Demo Patient").within(() => {
    cy.get("button")
      .filter(":visible")
      .then(($buttons) => {
        const chatButton = [...$buttons].find((button) => {
          const text = button.innerText
            .replace(/\s+/g, " ")
            .trim();

          const ariaLabel =
            button.getAttribute("aria-label") || "";

          const title =
            button.getAttribute("title") || "";

          return (
            /^Chat/i.test(text) ||
            /open chat|chatbot|chat assistant/i.test(ariaLabel) ||
            /open chat|chatbot|chat assistant/i.test(title)
          );
        });

        expect(
          chatButton,
          "red Chat launcher button"
        ).to.exist;

        cy.wrap(chatButton)
          .scrollIntoView()
          .should("be.visible")
          .click();
      });
  });

  // Confirm that clicking Chat opened the assistant.
  getStreamlitFrameBody("Suggested questions").within(() => {
    cy.contains("Suggested questions", {
      timeout: 30000
    }).should("be.visible");

    cy.contains("Language and settings")
      .should("be.visible");
  });
}