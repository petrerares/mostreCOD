/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import java.util.Random;
import java.util.ArrayList;
/**
 *
 * @author Petre
 */
public class JokeManager {
    private ArrayList<String> jokes;
   
    public JokeManager(){
        this.jokes=new ArrayList<>();
        
    }
    
    public void addJoke(String joke){
        this.jokes.add(joke);
    }
    
    public String drawJoke() {
        if (jokes.isEmpty()) {
            return "Jokes are in short supply.";
        }
        return jokes.get(new Random().nextInt(jokes.size()));
    }
    
    public void printJokes(){
        for(String x:this.jokes){
            System.out.println(x);
        }
    }
}
