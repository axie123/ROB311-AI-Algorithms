from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE


def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """

    ### START: Your code
    inferred = {}  # Dictionary to keep track of all of my inferred states.
    for symbol in symbols_list:
        inferred[symbol] = False # All are False at the beginning.

    agenda = deque()  # The agenda keeps track of the symbols that are in the input.
    for symbol in known_symbols:
        agenda.append(symbol)

    count = {}  # This dictionary keeps track of the length of the clauses.
    for clause in KB_clauses:
        count[str(clause.body)] = len(clause.body)

    while agenda:
        p = agenda.popleft()  # Take out the first symbol.
        if p == query:  # Check to see if the symbol is the target query. If so, return True.
            return True
        if inferred[p] == False:
            inferred[p] = True  # Mark the inferred symbol as True.
            for c in KB_clauses:
                if p in c.body:  # If the symbol is part of the clause:
                    count[str(c.body)] -= 1  # Reduce the length count of the clause to 1, as to remove the symbol from consideration.
                    if count[str(c.body)] == 0:  # If all of the clause has been found to contain a symbol in the agenda, the conclusion of the clause is loaded.
                        agenda.append(c.conclusion)
    return False  # If no match was found with the query, False is returned.
    ### END: Your code


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1,2,9,4,5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9,4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
