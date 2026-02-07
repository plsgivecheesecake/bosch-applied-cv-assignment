import streamlit as st


def main():
    """
    This is the main entrypoint of the streamlit web application, and
    is also responsible for displaying the contents of the home page.
    """
    st.set_page_config(page_title="Hello", page_icon=":material/sentiment_excited:")
    st.write(
        """
            # Hello! :material/sentiment_excited:
            
            Thank you for taking the time to review my submission.
            More details to follow soon.
        """
    )


if __name__ == "__main__":
    main()
