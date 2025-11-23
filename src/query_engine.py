def interactive_query(collection):
    print("\n--- QUERY ENGINE STARTED ---")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            # 1. Get user input
            question = input("Enter your question: ").strip()
            
            # 2. Check for exit command
            if question.lower() == 'exit':
                print("[Exited interactive mode]")
                break  # Exit the while loop
                
            # 3. Skip empty questions
            if not question:
                continue

            # 4. Query the collection - MUST BE INSIDE try block
            results = collection.query(query_texts=[question], n_results=3)
            
            # 5. Display results
            print(f"\n✓ Found {len(results['documents'][0])} results:\n")
            for i, doc in enumerate(results["documents"][0], 1):
                print(f"--- Result {i} ---")
                print(f"{doc}\n")
         
        except EOFError:
            # Handles Ctrl+D/Ctrl+Z input
            print("\n[Exited interactive mode]")
            break
        except KeyboardInterrupt:
            # Handles Ctrl+C
            print("\n[Interrupted by user]")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}\n")
        results = collection.query(query_texts=[question], n_results=3)
        for i, doc in enumerate(results["documents"][0]):
            print(f"\n--- Result {i+1} ---\n{doc}")  