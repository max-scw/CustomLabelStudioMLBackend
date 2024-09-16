import secrets
import string


def generate_unique_id(length: int = 10) -> str:
    """
    Generate a unique identifier of a specified length.
    Args:
        length (int): The length of the identifier. Defaults to 10.
    Returns:
        str: A unique identifier.
    """
    characters = string.ascii_letters + string.digits + '_'
    return ''.join(secrets.choice(characters) for _ in range(length))
