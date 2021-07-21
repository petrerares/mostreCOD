/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
/**
 *
 * @author Petre
 */
public class TextUI {
    private final Scanner scanner;
    private final SimpleDictionary dic;
    
    public TextUI(Scanner scanner,SimpleDictionary obj){
        this.scanner=scanner;
        this.dic=obj;
    }
    
    public void start(){
        while(true){
            System.out.println("Command: ");
            String cmd=this.scanner.nextLine();
            if(cmd.equals("end")){
                System.out.println("Bye Bye!");
                break;
            } else if(cmd.equals("add")){
                System.out.println("Word: ");
                String word=this.scanner.nextLine();
                System.out.println("Translation: ");
                String tran=this.scanner.nextLine();
                this.dic.add(word, tran);
            } else if(cmd.equals("search")){
                System.out.println("To be translated: ");
                String word=this.scanner.nextLine();
                String Trans=this.dic.translate(word);
                 if (Trans == null) {
                    System.out.print(MessageFormat.format("Word {0} was not found", word));
                } else {
                    System.out.println(Trans);
                }
            }  else {
                System.out.println("Unknown command");
            }
            
        }
        
        
    }
}
